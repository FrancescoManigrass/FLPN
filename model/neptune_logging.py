


def log_loss_training(opt,loss,step=None):
    for i in loss.keys():
        if step == None:
            opt.neptune.log_metric(i,loss[i])
        else:
            opt.neptune.log_metric(i, loss[i] / step)
