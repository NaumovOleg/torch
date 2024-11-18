import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def activate_function(val):
    return 1 if val >= 0.50 else 0


def main(house, rock, attr):
    x = torch.tensor([house, rock, attr], dtype=torch.float32, device=device)
    weights = torch.tensor([[0.3, 0.3, 0], [0.4, -0.5, 1]])

    # Hiddeden layer
    hiddenLayer = torch.mv(weights, x)
    print(f"Summ neirons in hidden layer {hiddenLayer}")

    Uh = torch.tensor([activate_function(x) for x in hiddenLayer], dtype=torch.float32)
    weightsOutput = torch.tensor([-1.0, 1.0])
    Zout = torch.dot(weightsOutput, Uh)

    Uh.to(device)

    print(
        {"hiddenLayer": hiddenLayer, "x": x, "Uh": Uh, "weightsOutput": weightsOutput}
    )
    result = activate_function(Zout)
    print(f"result {result} {Zout}")


main(1, 0, 1)
