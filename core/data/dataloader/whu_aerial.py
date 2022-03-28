import os
from torchvision import transforms
import cv2
import albumentations as A

class AerialSegmentation(object):

    NUM_CLASS = 2

    def __init__(self, root='/XXXXX/', split='train', mode=None, base_size=256,crop_size=256):
        super(AerialSegmentation, self).__init__()

        self.mode = mode
        self.crop_size = crop_size
        self.base_size = base_size
        _splits_dir = root
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
            _mask_dir = os.path.join(root, 'train', 'label')
            _image_dir = os.path.join(root, 'train', 'image')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
            _mask_dir = os.path.join(root, 'val', 'label')
            _image_dir = os.path.join(root, 'val', 'image')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
            _mask_dir = os.path.join(root, 'test', 'label')
            _image_dir = os.path.join(root, 'test', 'image')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n'))
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n'))
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _image_dir))

        ###################################################################
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.3),
            A.RandomResizedCrop(self.crop_size, self.crop_size, scale=(0.75, 1.5), p=0.5)
        ])

        self.train_transform_img = A.Compose([
            A.HueSaturationValue(10, 5, 10, p=0.3),
            A.GaussNoise(p=0.3)
        ])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.441, .449, .415], [.199, .187, .196]),  # aerial
        ])
        self.transform_mask = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.441, .449, .415], [.199, .187, .196]),  # aerial
            # transforms.Normalize([.443, .446, .409], [.182, .170, .177]),#aerial
        ])
    def __getitem__(self, index):
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # synchronized transform
        if self.mode == 'train':
            data = {"image": image,  "mask": mask}
            sample = self.train_transform(**data)
            image, mask = sample["image"],sample["mask"]

            data = {"image": image}
            sample_image = self.train_transform_img(**data)
            image = sample_image["image"]

            image = self.transform(image)
            mask = self.transform_mask(mask)
        elif self.mode == 'val':
            image = self.transform(image)
            mask = self.transform_mask(mask)
        elif self.mode == 'test':
            image = self.test_transform(image)
            mask = self.transform_mask(mask)
        mask = mask.long()[0]
        return image, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)


    @property
    def classes(self):
        """Category names."""
        return ('buildings', 'others', )


if __name__ == '__main__':
    dataset = AerialSegmentation()