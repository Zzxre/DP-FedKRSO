import torch


def aggregate_models_normal(global_model, client_models, args):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if args.lora_r > 0:
            if "lora" in k:  # Only aggregate LoRA parameters
                global_dict[k] = torch.stack(
                    [client_models[i][k].float() for i in range(len(client_models))], 0
                ).mean(0)

            if "classifier" in k:
                global_dict[k] = torch.stack(
                    [client_models[i][k].float() for i in range(len(client_models))], 0
                ).mean(0)
        else:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_ffa(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        if "lora_B" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_RSO(global_model, client_models, args):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():

        if "lora" in k:  # Only aggregate LoRA B parameters
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        elif "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

        elif "roberta.encoder.layer." in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_flora(global_model, client_models, args):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack(
            [client_models[i][k].float() for i in range(len(client_models))], 0
        ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model


def aggregate_models_flexlora(global_model, client_models, args):
    global_dict = global_model.state_dict()
    tmp_dict = {}
    for k in global_dict.keys():
        if "lora" in k:  # Only aggregate LoRA parameters
            if "lora_A" in k:
                tmp_dict[k] = torch.concat(
                    [client_models[i][k].float() for i in range(len(client_models))], 0
                )
            if "lora_B" in k:
                tmp_dict[k] = (torch.concat(
                    [client_models[i][k].float() for i in range(len(client_models))], 1
                ) @ tmp_dict[k]) / len(client_models)

                u, s, v = torch.svd(tmp_dict[k])
                global_dict[k] = u[:, args.lora_r] @ torch.diag(s[:args.lora_r].sqrt())
                global_dict[k.replace('lora_B', 'lora_A')] = torch.diag(s[:args.lora_r].sqrt()) @ v[:, :args.lora_r].T

        if "classifier" in k:
            global_dict[k] = torch.stack(
                [client_models[i][k].float() for i in range(len(client_models))], 0
            ).mean(0)

    global_model.load_state_dict(global_dict, strict=False)

    return global_model
