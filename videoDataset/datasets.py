import torch
from torch.utils.data import Dataset 
import cv2
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader
__all__ = ['VideoDataset', 'VideoLabelDataset']
from torchvision import utils
class VideoDataset(Dataset):
    """ Video Dataset for loading video.
        It will output only path of video (neither video file path or video folder path). 
        However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        Your video dataset can be image frames or video files.

    Args:
        csv_file (str): path fo csv file which store path of video file or video folder.
            the format of csv_file should like:
            
            # example_video_file.csv   (if the videos of dataset is saved as video file)

            path
            ~/path/to/video/file1.mp4
            ~/path/to/video/file2.mp4
            ~/path/to/video/file3.mp4
            ~/path/to/video/file4.mp4

            # example_video_folder.csv   (if the videos of dataset is saved as image frames)
            
            path
            ~/path/to/video/folder1/
            ~/path/to/video/folder2/
            ~/path/to/video/folder3/
            ~/path/to/video/folder4/

    Example:

        if the videos of dataset is saved as video file

        >>> import torch
        >>> from datasets import VideoDataset
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_file.csv",
        >>>     transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos in data_loader:
        >>>     print(videos.size())

        if the video of dataset is saved as frames in video folder
        The tree like: (The names of the images are arranged in ascending order of frames)
        ~/path/to/video/folder1
        ├── frame-001.jpg
        ├── frame-002.jpg
        ├── frame-003.jpg
        └── frame-004.jpg

        >>> import torch
        >>> from datasets import VideoDataset
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_folder.csv",
        >>>     transform = transforms.VideoFolderPathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos in data_loader:
        >>>     print(videos.size())
    """
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video """
        video = self.dataframe.iloc[index].path
        if self.transform:
            video = self.transform(video)
        return video


class VideoLabelDataset(Dataset):
    """ Dataset Class for Loading Video.
        It will output path and label. However, you can load video as torch.Tensor (C x L x H x W).
        See below for an example of how to read video as torch.Tensor.
        You can load tensor from video file or video folder by using the same way as VideoDataset.

    Args:
        csv_file (str): path fo csv file which store path and label of video file (or video folder).
            the format of csv_file should like:
            
            path, label
            ~/path/to/video/file1.mp4, 0
            ~/path/to/video/file2.mp4, 1
            ~/path/to/video/file3.mp4, 0
            ~/path/to/video/file4.mp4, 2

    Example:
        >>> import torch
        >>> import transforms
        >>> dataset = VideoDataset(
        >>>     "example_video_file_with_label.csv",
        >>>     transform = transforms.VideoFilePathToTensor()  # See more options at transforms.py
        >>> )
        >>> data_loader = torch.utils.data.DataLoader(dataset, batch_size = 1, shuffle = True)
        >>> for videos, labels in data_loader:
        >>>     print(videos.size())
    """
    def __init__(self, csv_file, is_train = True, transform=None):
        df = pd.read_csv(csv_file)
        self.dataframe = df.loc[df['is_train'] == 1]
        self.transform = transform 

    def __len__(self):
        """
        Returns:
            int: number of rows of the csv file (not include the header).
        """
        return len(self.dataframe)

    def __getitem__(self, index):
        """ get a video and its label """
        video = self.dataframe.iloc[index].path
        label = self.dataframe.iloc[index].label 
        # person = self.dataframe.iloc[index].person
        # print(video, label)
        if self.transform:
            video = self.transform(video)
        # return video, person, label #only for test with person emotion
        return video, label


if __name__ == '__main__':
    import torchvision
    import PIL 
    import transforms_param, transforms

    # # test for VideoDataset
    # dataset = VideoDataset(
    #     './data/example_video_file.csv', 
    # )
    # path = dataset[0]
    # print(path)

    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    # for video in test_loader:
    #     print(video)
    
    # test for VideoLabelDataset
    video_folder2tensor = transforms_param.VideoFolderPathToTensor(max_len = 3, padding_mode = 'last')
    trainset = VideoLabelDataset(
        '/projects/totem/sandeep/MEAD/15M15F_person_lab.csv', 
        transform = torchvision.transforms.Compose([
                           video_folder2tensor,
                           transforms.VideoResize([128, 128] ), #videos are normalized here so regardless of resize or not, this has to be called
                        ]))

    print(len(trainset), 'size of trainset')
    train_loader = DataLoader(trainset,
                      batch_size=1,
                      num_workers=4, 
                      shuffle=True,
                      drop_last = True, 
                      pin_memory=True
                      )

    dataset_size = len(train_loader.dataset)

    print('# training videos = %d' % dataset_size)
    
    person = [2, 16, 10, 18, 8, 19, 9, 20, 3, 19, 10, 29, 6, 21, 14, 28]
    emo = [0,  0,  1,  1, 2,  2, 3,  3, 4,  4,  5,  5, 6,  6,  7,  7]
    loop = True
    cc = 0
    while(loop):
        
        if cc==16:
            break;
        vdata, p_id, label = next(iter(train_loader))
        print(vdata[0].shape, label)
        for i in range(len(person)):
            if label == emo[i] and p_id == person[i]:
                print("found", p_id, label)
                cc = cc+1

                data = vdata[0]
                samplem1 = data[:,:,0,:,:]
                sample0  = data[:,:,1,:,:]
                samplep1 = data[:,:,2,:,:]
                i = 0
                utils.save_image(
                    samplem1,
                    './'+str(p_id.data[0].item())+"_"+str(label.data[0].item())+"_0.png",
                    nrow=int(1 ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    sample0,
                    './'+str(p_id.data[0].item())+"_"+str(label.data[0].item())+"_1.png",
                    nrow=int(1 ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )
                utils.save_image(
                    samplep1,
                    './'+str(p_id.data[0].item())+"_"+str(label.data[0].item())+"_2.png",
                    nrow=int(1 ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )


    # print("after the increment")
    # video_folder2tensor.update_offset(4)
    # i = 1
    # vdata, person, label = next(iter(train_loader))
    # data = vdata[0]
    # samplem1 = data[:,:,0,:,:]
    # sample0  = data[:,:,1,:,:]
    # samplep1 = data[:,:,2,:,:]
    # utils.save_image(
    #     samplem1,
    #     './'+f"/x{str(i).zfill(6)}_0.png",
    #     nrow=int(1 ** 0.5),
    #     normalize=True,
    #     range=(-1, 1),
    # )
    # utils.save_image(
    #     sample0,
    #    './'+f"/x{str(i).zfill(6)}_1.png",
    #     nrow=int(1 ** 0.5),
    #     normalize=True,
    #     range=(-1, 1),
    # )
    # utils.save_image(
    #     samplep1,
    #     './'+f"/x{str(i).zfill(6)}_2.png",
    #     nrow=int(1 ** 0.5),
    #     normalize=True,
    #     range=(-1, 1),
    # )
    # print(vdata[0].shape, vdata[1], vdata[2], label)



    # frame1 = torchvision.transforms.ToPILImage()(video[:, 29, :, :])
    # frame2 = torchvision.transforms.ToPILImage()(video[:, 39, :, :])
    # print(frame1.shape)
    # frame1.show()
    # frame2.show()

    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # for videos, labels in test_loader:
    #     print(videos.size(), label)

