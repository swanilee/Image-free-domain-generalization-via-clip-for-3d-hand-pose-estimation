
def unfrozen_param(selected_model):
    print('UNFROZEN layer start')
    print("Unfrozen Parameter index....")
    
    for idx, param in enumerate(selected_model.parameters()):
        param.requires_grad = True
        # print(idx, param.name, param.shape)
        print("Param unfrozen . . . ")


def freeze_param(selected_model):
    print('FREEZE layer start~!')
    print("Freeze Parameter index....")
    
    for idx, param in enumerate(selected_model.parameters()):
        param.requires_grad = False
        # print(idx, param.name, param.shape)
        print("Param freeze . . .")

