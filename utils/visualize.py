import cv2
from PIL import Image
import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import ast
from joblib import Parallel, delayed
num_worker = 32

def color_mask(mask, color, alpha_mask):
	'''
	mask: prediction mask [r, g, b, alpha]
	color: color of overlay image
	alpha_mask: binary  mask
	'''
	mask[alpha_mask] = color
	mask[~alpha_mask] = (0, 0, 0, 0)
	return mask
    

def visualize(img_dir, pred_dir, out_dir):
	alpha_blend = 0.6 
	threshold = 0.5

	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for img_name in tqdm(sorted(os.listdir(img_dir))):
		mask_name = os.path.splitext(img_name)[0] + '.png'
		img_path = os.path.join(img_dir, img_name)
		mask_path = os.path.join(pred_dir, mask_name)
		out_path = os.path.join(out_dir, mask_name)

		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

		mas = cv2.imread(mask_path)
		mas = cv2.cvtColor(mas, cv2.COLOR_BGR2RGB)
		mas = cv2.cvtColor(mas, cv2.COLOR_RGB2RGBA)

		mas_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		mas_gray = mas_gray > (255 * threshold)
		
		mas = color_mask(mas, (0, 0, 255, 255 * alpha_blend), mas_gray) # default color: blue (0, 0, 255)
		merged = Image.alpha_composite(Image.fromarray(img), Image.fromarray(mas))
		
		merged.save(out_path)

def visualize_sem(pred_dir, out_dir):
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	color_map = pd.read_csv('utils/GSD-S_color_map.csv')
	color_map_rgb = np.array([np.array(ast.literal_eval(c)) for c in color_map['rgb'].tolist()])

	# print(f'sorted(os.listdir(pred_dir)) {len(sorted(os.listdir(pred_dir)))} {sorted(os.listdir(pred_dir))[0]}')

	for img_name in tqdm(sorted(os.listdir(pred_dir))):
		seseg = Image.open(os.path.join(pred_dir, img_name))
		# print(seseg.shape, np.unique(seseg), seseg);input('...')
		seseg_color = (color_map_rgb[np.array(seseg)]).astype(np.uint8)
		seseg_color = Image.fromarray(seseg_color, 'RGB')
		seseg_color.save(os.path.join(out_dir, img_name))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image", type=str, required=True)
	parser.add_argument("-p", "--prediction", type=str, required=True)
	parser.add_argument("-o", "--output", type=str, required=True)
	parser.add_argument("-s", "--semantic", action='store_true')

	args = parser.parse_args()

	print('Visualizing prediction masks:')
	visualize(args.image, os.path.join(args.prediction,'output'), os.path.join(args.output,'output_visualized'))

	if args.semantic:
		print('Visualizing semantic segmentations:')
		visualize_sem(os.path.join(args.prediction,'semantic'), os.path.join(args.output,'semantic_colored'))
