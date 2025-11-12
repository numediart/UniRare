import os 
from . import Unisal
from . import TranSalNetDense
from . import TranSalNetRes
from . import TempSal

import torch


def load_model(args , DEFAULT_DEVICE):
    path_ = os.path.dirname(os.path.abspath(__file__))
    print("Loading model from: ", path_)

    if args.model == "Unisal":
        model = Unisal(
            bypass_rnn=False,
        )

        if os.path.exists(path_ + "/Unisal/weights/" + "weights_best.pth"):
            print("Model Load")
            model.load_weights(path_ + "/Unisal/weights/" + "weights_best.pth")
        else:
            print("Model not found")


    elif args.model == "TranSalNetDense":
        model = TranSalNetDense()
        model.load_state_dict(torch.load(path_ + '/TranSalNet/weights/TranSalNet_Dense.pth', map_location=DEFAULT_DEVICE))

    elif args.model == "TranSalNetRes":
        model = TranSalNetRes()
        model.load_state_dict(torch.load(path_ + '/TranSalNet/weights/TranSalNet_Res.pth', map_location=DEFAULT_DEVICE))

    elif args.model == "TempSal":
        
        model_checkpoint_path= path_ + "/TempSal/weights/multilevel_tempsal.pt"
        model = TempSal(
            device=DEFAULT_DEVICE,
            model_path=model_checkpoint_path,
            model_vol_path= model_checkpoint_path,
            time_slices=5,
            train_model=0
        )

    return model
