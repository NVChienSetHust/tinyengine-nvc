def find_previous_link_op(model, target_op):
    tensor_name = target_op["inputs"][0]["name"]
    for idx, previous_op in enumerate(model):
        if previous_op["outputs"][0]["name"] == tensor_name:
            return idx, previous_op


def find_previous_link_op_input2(model, target_op):
    tensor_name = target_op["inputs"][1]["name"]
    for idx, previous_op in enumerate(model):
        if previous_op["outputs"][0]["name"] == tensor_name:
            return idx, previous_op


def find_following_link_op(model, target_op):
    tensor_name = target_op["outputs"][0]["name"]
    for idx, following_op in enumerate(model):
        for input_t in following_op["inputs"]:
            if input_t["name"] == tensor_name:
                return idx, following_op
    return None, None


def reorderGroupConv_TransponseConv(model):
    global_index = 0
    # compact the group conv op ordering
    # cast -> reshape -> (... which we want to skip) -> tile -> reshape -> nn.conv2d -> reshape -> sum ->
    # transpose -> [max -> divide -> divide (int8 bp)]
    for _, op in enumerate(model):
        if op["type"] == "cast":
            resshape_idx, reshape = find_following_link_op(model, op)
            if not reshape:
                continue
            if reshape["type"] != "reshape":
                continue
            conv2d_idx, conv2d = find_following_link_op(model, reshape)
            if reshape["type"] == "reshape" and conv2d["type"] == "nn.conv2d":
                resshape2_idx, reshape2 = find_previous_link_op_input2(model, conv2d)
                tile_idx, tile = find_previous_link_op(model, reshape2)
                if not (tile["type"] == "tile" and reshape2["type"] == "reshape"):
                    continue
                model.remove(reshape)
                model.remove(op)
                model.insert(tile_idx - 2, op)
                model.insert(tile_idx - 1, reshape)
    # compact the transpose conv
    while global_index < len(model):
        # find cast - > reshape -> tile ...-> conv2d(group) -> ... -> transpose (wiht 'weight' in 'output_info)
        conv2d_set_start_idx = None
        conv2d_set_end_idx = None
        transpose_conv_idx = None
        for cnt in range(global_index, len(model)):
            op = model[cnt]
            if op["type"] == "transpose" and "meta" in op["outputs"][0] and "output_info" in op["outputs"][0]["meta"]:
                conv2d_set_end_idx = cnt
                # back trace to the conv2d
                for back_inx in range(global_index, cnt):
                    back_op = model[back_inx]
                    if back_op["type"] == "nn.conv2d":
                        groups = back_op["attrs"]["groups"]
                        input_ch = back_op["inputs"][0]["shape"][1]
                        output_ch = back_op["outputs"][0]["shape"][1]
                        if not (input_ch == groups == output_ch):  # pylint: disable=C0325
                            conv2d_set_start_idx = back_inx
                            break
                if conv2d_set_start_idx is not None:
                    # find the closest cast
                    conv2d_set_start_idx, cast_op = find_previous_link_op(model, model[conv2d_set_start_idx])
                    while cast_op["type"] != "cast":
                        conv2d_set_start_idx, cast_op = find_previous_link_op(model, model[conv2d_set_start_idx])
                    break
        if conv2d_set_end_idx is None:
            break
        # find the closest transpose conv 2d -> ... -> sum after transpose
        for cnt in range(conv2d_set_end_idx, len(model)):
            if model[cnt]["type"] == "nn.conv2d_transpose":
                transpose_conv_idx = cnt
                # find the closest sum
                transpose_conv_idx, sum_op = find_following_link_op(model, model[transpose_conv_idx])
                # case 1. reaching the sum, this means the calculation cycle of this transpose conv is finished
                while sum_op["type"] != "sum":
                    transpose_conv_idx, sum_op = find_following_link_op(model, model[transpose_conv_idx])
                break
        # no more subgraphs to reroder
        if None in [conv2d_set_start_idx, conv2d_set_end_idx, transpose_conv_idx]:
            break
        # update the global index
        # global_index = cnt
        # reoder these two parts
        if not (None in [conv2d_set_start_idx, conv2d_set_end_idx, transpose_conv_idx]):
            new_model_first = model[0:conv2d_set_start_idx]
            new_model_second = model[conv2d_set_start_idx : conv2d_set_end_idx + 1]
            new_model_thrid = model[conv2d_set_end_idx + 1 : transpose_conv_idx + 1]
            new_model_final = model[transpose_conv_idx + 1 :]

            model = []
            model += new_model_first
            model += new_model_thrid
            model += new_model_second
            global_index = len(model)
            model += new_model_final

    return model


def reorderGroupConv_TransponseConv_int8(model):
    global_index = 0
    # compact the group conv op ordering
    # compact the transpose conv

    while global_index < len(model):
        conv2d_set_start_idx = None
        conv2d_set_end_idx = None
        transpose_conv_idx = None
        for cnt in range(global_index, len(model)):
            op = model[cnt]
            # Group conv: reshape -> (... which we want to skip) -> tile -> reshape ->
            #  nn.conv2d -> reshape -> sum -> transpose ->
            # [abs -> max -> divide -> divide -> cast (int8 bp)]
            if op["type"] == "cast" and "meta" in op["outputs"][0] and "output_info" in op["outputs"][0]["meta"]:
                conv2d_set_end_idx = cnt
                # back trace to the conv2d/transpose conv2d
                for back_inx in range(global_index, cnt):
                    back_op = model[back_inx]
                    # if back_op["type"] == "nn.conv2d_transpose":
                    # raise NotImplementedError
                    if back_op["type"] == "nn.conv2d":
                        groups = back_op["attrs"]["groups"]
                        input_ch = back_op["inputs"][0]["shape"][1]
                        output_ch = back_op["outputs"][0]["shape"][1]
                        if not (input_ch == groups == output_ch):  # pylint: disable=C0325
                            conv2d_set_start_idx = back_inx
                            break
                if conv2d_set_start_idx is not None:
                    # find the closest reshape
                    conv2d_set_start_idx, reshape_op = find_previous_link_op(model, model[conv2d_set_start_idx])
                    while reshape_op["type"] != "reshape":
                        conv2d_set_start_idx, reshape_op = find_previous_link_op(model, model[conv2d_set_start_idx])
                    break
        if conv2d_set_end_idx is None:
            break
        # find the closest transpose conv 2d -> ... -> sum after transpose
        for cnt in range(conv2d_set_end_idx, len(model)):
            if model[cnt]["type"] == "nn.conv2d_transpose":
                transpose_conv_idx = cnt
                # find the closest sum
                transpose_conv_idx, sum_op = find_following_link_op(model, model[transpose_conv_idx])
                # case 1. reaching the sum, this means the calculation cycle of this transpose conv is finished
                while sum_op["type"] != "sum":
                    transpose_conv_idx, sum_op = find_following_link_op(model, model[transpose_conv_idx])
                break
        # no more subgraphs to reroder
        if None in [conv2d_set_start_idx, conv2d_set_end_idx, transpose_conv_idx]:
            break
        # update the global index
        # global_index = cnt
        # reoder these two parts
        if not (None in [conv2d_set_start_idx, conv2d_set_end_idx, transpose_conv_idx]):
            new_model_first = model[0:conv2d_set_start_idx]
            new_model_second = model[conv2d_set_start_idx : conv2d_set_end_idx + 1]
            new_model_thrid = model[conv2d_set_end_idx + 1 : transpose_conv_idx + 1]
            new_model_final = model[transpose_conv_idx + 1 :]

            model = []
            model += new_model_first
            model += new_model_thrid
            model += new_model_second
            global_index = len(model)
            model += new_model_final

    return model

def reorderAvgPool2dGradFilt(model):
    global_index = 0
    while global_index < len(model):
        cast_idx = None
        avg_pool2d_idx = None
        mul_idx = None
        mcutruncate_idx = None
        second_cast_idx = None
        mcuconv2davg_idx = None

        # Find the pattern: cast -> avg_pool2d -> mul -> mcutruncate -> cast
        for i in range(global_index, len(model)):
            op = model[i]
            if op["type"] == "cast":
                cast = op
                cast_idx = i
                avg_pool2d_idx, avg_pool2d = find_following_link_op(model, op)
                if not avg_pool2d:
                    continue
                if avg_pool2d["type"] != "nn.avg_pool2d":
                    continue
                print("------------------------------found cast -> avgpool GF pattern--------------------------------")
                mul_idx, mul = find_following_link_op(model, avg_pool2d)
                print(avg_pool2d_idx)
                if not mul:
                    continue
                if mul["type"] != "multiply":
                    continue
                if mul["type"] == "multiply":
                    print("------------------------------found cast -> avgpool -> mul GF pattern--------------------------------")
                mcutruncate_idx, mcutruncate = find_following_link_op(model, mul)
                if not mcutruncate:
                    continue
                if mcutruncate["type"] != "nn.mcutruncate":
                    continue
                if mcutruncate["type"] == "nn.mcutruncate":
                    print("------------------------------found cast -> avgpool -> mul -> truncate GF pattern--------------------------------")
                second_cast_idx, second_cast = find_following_link_op(model, mcutruncate)
                if not second_cast:
                    continue
                if second_cast["type"] != "cast":
                    continue
                # Find the closest mcuconv2davg BEFORE the cast -> avg_pool2d
                if avg_pool2d["type"] == "nn.avg_pool2d" and mul["type"] == "multiply" and mcutruncate["type"] == "nn.mcutruncate" and second_cast["type"] == "cast":
                    print("------------------------------found avgpool GF pattern--------------------------------")
                    mcuconv2davg_idx, mcuconv2davg = find_previous_link_op(model, op)
                    # Reorder the sequence
                    # Move the sequence cast -> avg_pool2d right after mcuconv2davg
                    # Remove the cast and avg_pool2d operations
                    model.remove(cast)
                    model.remove(avg_pool2d)  # Remove avg_pool2d first because it's later in the list
                    model.remove(mul)
                    model.remove(mcutruncate)
                    model.remove(second_cast)

                    # Insert cast -> avg_pool2d after mcuconv2davg
                    model.insert(mcuconv2davg_idx + 1, cast)
                    model.insert(mcuconv2davg_idx + 2, avg_pool2d)
                    model.insert(mcuconv2davg_idx + 3, mul)
                    model.insert(mcuconv2davg_idx + 4, mcutruncate)
                    model.insert(mcuconv2davg_idx + 5, second_cast)
                else:
                    print("------------------------------do not found avgpool GF pattern--------------------------------")
        # If we didn't find the full pattern, exit the loop
        if None in [cast_idx, avg_pool2d_idx, mul_idx, mcutruncate_idx, second_cast_idx]:
            break

        # If no mcuconv2davg was found, continue
        if mcuconv2davg_idx is None:
            global_index = second_cast_idx + 1
            continue

        # Update the global index to continue from where we left off
        global_index = second_cast_idx + 1

    return model