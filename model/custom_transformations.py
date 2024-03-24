import torch


def crop_images(input_v, boxes, opt):
    ##############
    bs = boxes.shape[0]

    source_width = torch.tensor(224)
    source_height = torch.tensor(224)
    output_width = torch.tensor(224)
    output_height = torch.tensor(224)
    theta = torch.zeros(bs, 2, 3)
    if opt.cuda:
        source_width = source_width.cuda()
        source_height = source_height.cuda()
        output_width = output_width.cuda()
        output_height = output_height.cuda()
        theta = theta.cuda()

    target_y0 = boxes[:, 1].float()
    target_x0 = boxes[:, 0].float()
    target_y1 = boxes[:, 3].float()
    target_x1 = boxes[:, 2].float()

    theta[:, 0, 0] = (target_x1 - target_x0) / (source_width - 1)
    theta[:, 0, 2] = (target_x1 + target_x0 - source_width + 1) / (source_width - 1)
    theta[:, 1, 1] = (target_y1 - target_y0) / (source_height - 1)
    theta[:, 1, 2] = (target_y1 + target_y0 - source_height + 1) / (source_height - 1)
    grid = torch.nn.functional.affine_grid(theta, (bs, 3, output_height, output_width))
    if opt.cuda:
        grid = grid.cuda()

    ###############
    # img0 = cv2.imread('1.jpg')
    # img1 = cv2.imread('2.jpg')
    # m0=img0[:source_height, :source_width, :]
    # m1=img1[:source_height, :source_width, :]
    # cv2.imwrite('m0.jpg', m0)
    # cv2.imwrite('m1.jpg', m1)
    # n1 = torch.from_numpy(np.transpose(img1, (2, 0, 1)))
    # n0 = torch.from_numpy(np.transpose(img0, (2, 0, 1)))
    # input_v = input_v.permute(0,3,1,2)

    # t = torch.zeros(bs, 3, source_height, source_width)
    # t[0] = n0
    # t[1] = n1
    # t = input_v

    ##############
    crp = torch.nn.functional.grid_sample(input_v, grid)
    """
    r0 = crp[0].cpu().detach().numpy()
    r1 = crp[1].cpu().detach().numpy()
    #r0 = np.transpose(r0, (1, 2, 0))
    #r1 = np.transpose(r1, (1, 2, 0))
    #cv2.imwrite('r0.jpg', r0)
    #cv2.imwrite('r1.jpg', r1)
    fig = plt.figure()
    plt.imshow(r0[0,:])
    fig.savefig('r0.png', dpi=fig.dpi)

    fig = plt.figure()
    plt.imshow(r1[0,:])
    fig.savefig('r1.png', dpi=fig.dpi)
    """

    return crp