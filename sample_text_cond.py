import os

from utils.util import seed_everything
from utils.util_3d import init_mesh_renderer, sdf_to_mesh
from utils.qual_util import load_bert2vq_model, get_lang_prob, save_mesh_as_gif
from utils.demo_util import get_shape_comp_model, get_shape_comp_opt, make_dummy_batch

seed_everything(111)

res_dir = 'results'
if not os.path.exists(res_dir): os.makedirs(res_dir)

gpu_id = 0
nimgs=6

opt = get_shape_comp_opt(gpu_id=gpu_id)
opt.dataset_mode = "shapenet_lang"
model = get_shape_comp_model(opt)
model.eval()

""" setup renderer """
dist, elev, azim = 1.7, 20, 20
mesh_renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=opt.device)


bert2vq = load_bert2vq_model(opt)

test_data = make_dummy_batch(nimgs)

text_conditional = "Couch with round arms"
lang_conditional_prob = get_lang_prob(bert2vq,text_conditional)
lang_conditional_prob = lang_conditional_prob.repeat(1, nimgs, 1)

topk =10
alpha = .5

model.inference(test_data, topk=topk, prob=lang_conditional_prob, alpha=alpha)
gen_mesh = sdf_to_mesh(model.x_recon_tf)

gen_gif_name = f'{res_dir}/lang-guided-gen.gif'
save_mesh_as_gif(mesh_renderer, gen_mesh, nrow=3, out_name=gen_gif_name)
