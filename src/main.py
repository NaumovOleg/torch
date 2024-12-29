# src/main.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import x_train, y_train

print(x_train, y_train)


def main():
    print("Training Data:")


if __name__ == "__main__":
    main()
