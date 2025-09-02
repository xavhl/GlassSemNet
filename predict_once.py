import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm
from joblib import Parallel, delayed
num_worker = 32

from model.GlassSemNet import GlassSemNet
from utils.dataloader import get_loader_testbatch
from utils.crf_refine import crf_refine

####### GLOBALS #######
ckpt_path = '../GlassSemNetv2.pth'

model = GlassSemNet()
ckpt_dict = torch.load(ckpt_path)
model.load_state_dict(ckpt_dict)
print('loaded model:',ckpt_path)
#######################

def save_pred(images_tensor, pred_tensor, original_size, save_path):
	
	res = pred_tensor.unsqueeze(0)
	res = res.sigmoid()
	res = (res - res.min()) / (res.max() - res.min() + 1e-8)

	predict_np = (res.squeeze().cpu().data.numpy() * 255).astype(np.uint8)
	predict_np = predict_np.copy(order='C')

	crf_input = images_tensor.squeeze().cpu().data.numpy() * 255
	crf_input = np.transpose(crf_input, (1, 2, 0)).astype(np.uint8)
	crf_input = crf_input.copy(order='C')
	predict_np = crf_refine(crf_input, predict_np)
	
	predict_np = np.where(predict_np<127.5, 0, 255).astype(np.uint8)

	im = Image.fromarray(predict_np)
	imo = im.resize(original_size)
	imo.save(save_path)

def save_pred_sem(pred_tensor, original_size, save_path):

	predict_np = (pred_tensor.unsqueeze(0).data.cpu().numpy().squeeze()).astype(np.uint8)
	
	im = Image.fromarray(predict_np)
	imo = im.resize(original_size, resample=Image.NEAREST) # resizing messes up predictions
	imo.save(save_path)
	
def predict(test_loader, model, save_dir, semantic):
	model.to(device)
	model.eval()

	for images, names, sizes in tqdm(test_loader):
		images = images.to(device)
		with torch.no_grad():
			output = model(images)
			preds = F.interpolate(output['out'], size=(images.shape[-2:]), mode='bilinear', align_corners=True)
			sem_preds = F.interpolate(output['semantic_pred'], size=(images.shape[-2:]), mode='bilinear', align_corners=True)
			sem_preds = torch.argmax(sem_preds, 1)

			for j in range(preds.shape[0]):
				original_size = (sizes[0][j], sizes[1][j])
				save_pred(images[j], preds[j], original_size, os.path.join(save_dir, 'output', names[j] ))
				if semantic:
					save_pred_sem(sem_preds[j], original_size, os.path.join(save_dir, 'semantic', names[j]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print('device:', device)

def run(image_root:str, save_dir:str, semantic:bool, batchsize:int=32):	
	global test_loader, model
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		
	test_loader = get_loader_testbatch(image_root, batchsize, trainsize=384)

	predict(test_loader, model, save_dir, semantic)

if __name__ == '__main__':

	run(image_root='../inputs', save_dir='../outputs', semantic=False)
