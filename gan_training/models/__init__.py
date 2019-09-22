from gan_training.models import (
    resnet, resnet2, connet, DCnet
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'connet': connet.Generator,
    'DCnet': DCnet.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'connet': connet.Discriminator,
    'DCnet': DCnet.Discriminator
}
