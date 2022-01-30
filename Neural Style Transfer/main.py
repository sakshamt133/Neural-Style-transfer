from content_img import content, img
from style_img import style
import torch
import utilis
from model import Model
from torchvision.utils import save_image


model = Model(utilis.layers)
content_out = model(content)
style_out = model(style)
generated_ = img.copy()
generated = utilis.image_process(generated_)
generated.requires_grad_(True)
opt = torch.optim.Adam([generated], lr=utilis.lr)


for epoch in range(utilis.epochs):
    generated_out = model(generated)
    content_loss = 0
    style_loss = 0

    for cont, sty, gen in zip(content_out, style_out, generated_out):
        content_loss += torch.mean((cont-gen)**2)
        style_ = sty.reshape(sty.shape[0], -1)
        style_mat = torch.einsum("ba, da -> bd", style_, style_)
        gen_ = gen.reshape(cont.shape[0], -1)
        gen_mat = torch.einsum("ba, da -> bd", gen_, gen_)
        style_loss += torch.mean((gen_mat - style_mat)**2)

    loss = utilis.alpha * content_loss + utilis.beta * style_loss
    print(f"for epoch {epoch} loss is {loss}")
    if epoch % 5 == 0:
        save_image(generated, "gen.png")
    loss.backward(retain_graph=True)
    opt.step()
    opt.zero_grad()
