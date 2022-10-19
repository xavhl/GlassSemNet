import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm

from model.GlassSemNet import GlassSemNet
from utils.dataloader import get_loader_testbatch
from utils.crf_refine import crf_refine

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

def predict(test_loader, model, save_dir, device):
	model.to(device)
	model.eval()

	for images, names, sizes in tqdm(test_loader):
		images = images.to(device)
		with torch.no_grad():
			preds = model(images)#[0] # pred: [output, aux1, aux2]
			preds = F.interpolate(preds, size=(images.shape[-2:]), mode='bilinear', align_corners=True)
			
			for j in range(preds.shape[0]):
				original_size = (sizes[0][j], sizes[1][j])
				save_pred(images[j], preds[j], original_size, save_dir+names[j])

def main(args):	
	if not os.path.exists(args.output):
		os.makedirs(args.output)

	ckpt_path = args.checkpoints # '/raid/home/yhyeung2/fyp/codes/lambdalabs_codes/trained_models/GlassSem_UpTo2_20220815_glass_seg/GlassSemNet.pth'
	image_root = args.input # '/raid/home/yhyeung2/fyp/datasets/whole_sesegs/test/images/'
	save_dir = args.output # '/raid/home/yhyeung2/fyp/codes/lambdalabs_codes/test_maps/GlassSem_UpTo2_20220815_glass_seg/whole_sesegs/epoch128_modelparams_test/output/'
	batchsize = args.batchsize

	test_loader = get_loader_testbatch(image_root, batchsize, trainsize=384)

	model = GlassSemNet()
	ckpt_dict = torch.load(ckpt_path)
	model.load_state_dict(ckpt_dict)
	print('loaded model:',ckpt_path)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print('device:', device)
	predict(test_loader, model, save_dir, device)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--checkpoints", type=str, required=True)
	parser.add_argument("-i", "--input", type=str, required=True)
	parser.add_argument("-o", "--output", type=str, required=True)
	parser.add_argument("-batch", "--batchsize", action="store_false", default=32)
	args = parser.parse_args()

	main(args)