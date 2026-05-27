import torchvision
import torchvision.transforms as transforms

MNIST_CLASSES = [str(i) for i in range(10)]
FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_datasets(dataset_name="MNIST", data_dir="./data"):
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_name == "MNIST":
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return train, test, MNIST_CLASSES
    else:
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        return train, test, FMNIST_CLASSES
