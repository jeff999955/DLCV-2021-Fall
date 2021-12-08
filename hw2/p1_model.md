Generator
Generator(
  (initial_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (blocks): ModuleList(
    (0): GeneratorBlock(
      (to_style1): Linear(in_features=512, out_features=512, bias=True)
      (to_noise1): Linear(in_features=1, out_features=512, bias=True)
      (conv1): Conv2DMod()
      (to_style2): Linear(in_features=512, out_features=512, bias=True)
      (to_noise2): Linear(in_features=1, out_features=512, bias=True)
      (conv2): Conv2DMod()
      (activation): LeakyReLU(negative_slope=0.2, inplace=True)
      (to_rgb): RGBBlock(
        (to_style): Linear(in_features=512, out_features=512, bias=True)
        (conv): Conv2DMod()
        (upsample): Sequential(
          (0): Upsample(scale_factor=2.0, mode=bilinear)
          (1): Blur()
        )
      )
    )
    (1): GeneratorBlock(
      (upsample): Upsample(scale_factor=2.0, mode=bilinear)
      (to_style1): Linear(in_features=512, out_features=512, bias=True)
      (to_noise1): Linear(in_features=1, out_features=256, bias=True)
      (conv1): Conv2DMod()
      (to_style2): Linear(in_features=512, out_features=256, bias=True)
      (to_noise2): Linear(in_features=1, out_features=256, bias=True)
      (conv2): Conv2DMod()
      (activation): LeakyReLU(negative_slope=0.2, inplace=True)
      (to_rgb): RGBBlock(
        (to_style): Linear(in_features=512, out_features=256, bias=True)
        (conv): Conv2DMod()
        (upsample): Sequential(
          (0): Upsample(scale_factor=2.0, mode=bilinear)
          (1): Blur()
        )
      )
    )
    (2): GeneratorBlock(
      (upsample): Upsample(scale_factor=2.0, mode=bilinear)
      (to_style1): Linear(in_features=512, out_features=256, bias=True)
      (to_noise1): Linear(in_features=1, out_features=128, bias=True)
      (conv1): Conv2DMod()
      (to_style2): Linear(in_features=512, out_features=128, bias=True)
      (to_noise2): Linear(in_features=1, out_features=128, bias=True)
      (conv2): Conv2DMod()
      (activation): LeakyReLU(negative_slope=0.2, inplace=True)
      (to_rgb): RGBBlock(
        (to_style): Linear(in_features=512, out_features=128, bias=True)
        (conv): Conv2DMod()
        (upsample): Sequential(
          (0): Upsample(scale_factor=2.0, mode=bilinear)
          (1): Blur()
        )
      )
    )
    (3): GeneratorBlock(
      (upsample): Upsample(scale_factor=2.0, mode=bilinear)
      (to_style1): Linear(in_features=512, out_features=128, bias=True)
      (to_noise1): Linear(in_features=1, out_features=64, bias=True)
      (conv1): Conv2DMod()
      (to_style2): Linear(in_features=512, out_features=64, bias=True)
      (to_noise2): Linear(in_features=1, out_features=64, bias=True)
      (conv2): Conv2DMod()
      (activation): LeakyReLU(negative_slope=0.2, inplace=True)
      (to_rgb): RGBBlock(
        (to_style): Linear(in_features=512, out_features=64, bias=True)
        (conv): Conv2DMod()
        (upsample): Sequential(
          (0): Upsample(scale_factor=2.0, mode=bilinear)
          (1): Blur()
        )
      )
    )
    (4): GeneratorBlock(
      (upsample): Upsample(scale_factor=2.0, mode=bilinear)
      (to_style1): Linear(in_features=512, out_features=64, bias=True)
      (to_noise1): Linear(in_features=1, out_features=32, bias=True)
      (conv1): Conv2DMod()
      (to_style2): Linear(in_features=512, out_features=32, bias=True)
      (to_noise2): Linear(in_features=1, out_features=32, bias=True)
      (conv2): Conv2DMod()
      (activation): LeakyReLU(negative_slope=0.2, inplace=True)
      (to_rgb): RGBBlock(
        (to_style): Linear(in_features=512, out_features=32, bias=True)
        (conv): Conv2DMod()
      )
    )
  )
  (attns): ModuleList(
    (0): None
    (1): None
    (2): None
    (3): None
    (4): Sequential(
      (0): Residual(
        (fn): PreNorm(
          (fn): LinearAttention(
            (nonlin): GELU()
            (to_q): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (to_kv): DepthWiseConv2d(
              (net): Sequential(
                (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (1): Conv2d(64, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
            )
            (to_out): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (norm): ChanNorm()
        )
      )
      (1): Residual(
        (fn): PreNorm(
          (fn): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.2, inplace=True)
            (2): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (norm): ChanNorm()
        )
      )
    )
  )
)
Discriminator
Discriminator(
  (blocks): ModuleList(
    (0): DiscriminatorBlock(
      (conv_res): Conv2d(3, 64, kernel_size=(1, 1), stride=(2, 2))
      (net): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (downsample): Sequential(
        (0): Blur()
        (1): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
    (1): DiscriminatorBlock(
      (conv_res): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2))
      (net): Sequential(
        (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (downsample): Sequential(
        (0): Blur()
        (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
    (2): DiscriminatorBlock(
      (conv_res): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2))
      (net): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (downsample): Sequential(
        (0): Blur()
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
    (3): DiscriminatorBlock(
      (conv_res): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
      (net): Sequential(
        (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (downsample): Sequential(
        (0): Blur()
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
    (4): DiscriminatorBlock(
      (conv_res): Conv2d(512, 512, kernel_size=(1, 1), stride=(2, 2))
      (net): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
      (downsample): Sequential(
        (0): Blur()
        (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
    (5): DiscriminatorBlock(
      (conv_res): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (net): Sequential(
        (0): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): LeakyReLU(negative_slope=0.2, inplace=True)
        (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): LeakyReLU(negative_slope=0.2, inplace=True)
      )
    )
  )
  (attn_blocks): ModuleList(
    (0): Sequential(
      (0): Residual(
        (fn): PreNorm(
          (fn): LinearAttention(
            (nonlin): GELU()
            (to_q): Conv2d(64, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (to_kv): DepthWiseConv2d(
              (net): Sequential(
                (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
                (1): Conv2d(64, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              )
            )
            (to_out): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (norm): ChanNorm()
        )
      )
      (1): Residual(
        (fn): PreNorm(
          (fn): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
            (1): LeakyReLU(negative_slope=0.2, inplace=True)
            (2): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
          )
          (norm): ChanNorm()
        )
      )
    )
    (1): None
    (2): None
    (3): None
    (4): None
    (5): None
  )
  (quantize_blocks): ModuleList(
    (0): None
    (1): None
    (2): None
    (3): None
    (4): None
    (5): None
  )
  (final_conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (flatten): Flatten()
  (to_logit): Linear(in_features=2048, out_features=1, bias=True)
)
