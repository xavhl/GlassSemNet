import cv2
from PIL import Image
import os
import argparse
from tqdm import tqdm

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

	img_list = sorted(os.listdir(img_dir))

	for img_name in tqdm(img_list):
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

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image", type=str, required=True)
	parser.add_argument("-p", "--prediction", type=str, required=True)
	parser.add_argument("-o", "--output", type=str, required=True)
	args = parser.parse_args()

	if not os.path.exists(args.output):
		os.makedirs(args.output)
	
	visualize(args.image, args.prediction, args.output)