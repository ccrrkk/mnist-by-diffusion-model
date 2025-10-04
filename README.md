# diffusion-model
diffusion model on mnist

forked from https://github.com/guchengzhong/diffusion-model

源代码使用了notebook, 在源代码的基础上我将其拆分成了多个文件，并参照[猛猿的博客](https://zhuanlan.zhihu.com/p/655568910)添加了注释

源代码采用的是DDPM，无条件生成，输出的数字类别是随机的，无法人为控制。为实现类别控制，我加入了conditional diffusion：在模型中增加类别embedding层，类别embedding通常与时间embedding相加后输入到UNet中, 实现类别信息与图像特征结合，并带标签进行训练，实现按类别生成数字。

如果希望进一步支持自然语言输入（如Stable Diffusion），则需通过预训练文本编码器（如CLIP或T5）将文本描述转为embedding，并在UNet各层通过cross-attention机制，将文本embedding与图像特征进行融合，实现根据文本生成高质量图像。与类别embedding的简单相加不同，文本embedding通过cross-attention动态影响图像生成过程。