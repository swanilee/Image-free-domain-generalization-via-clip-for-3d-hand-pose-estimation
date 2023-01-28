## Image free domain generalization via clip for 3d hand pose estimation

### Prerequisties
The code needs the following libraries:
* Python
* Anaconda 
* Pytorch 1.7.1


## Data Preparation
We use the dataset of FreiHAND for training, and STB, RHD for testing.

## Pretrained file
* Pretrained file Download : [Link Here](https://drive.google.com/drive/folders/1olYGUlt1pcoCC6I7wh4lC-MUESiseZgc?usp=sharing)
```
Path: ../weights/snapshot_45.pth.tar
```

## Testing

```
# If you change the dataset, change "config.py" in main directory.
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --master_port 12345 test.py --gpu 0 --stage lixel --test_epoch "45" --scale 1 --rot 0

```
