import torch



def create_fake(x_,false_values_for_row):
    mask=x_< torch.mean(x_, dim=1).view(-1, 1)
    #mask = torch.ones((x_.shape[0], x_.shape[1])).bool()
    #false_values_for_row = 15

    # Selezionare casualmente 15 valori True per ogni riga e impostarli a False nella nuova maschera
    for i in range(mask.size(0)):  # Per ogni riga
        true_indices = torch.where(mask[i])[0]  # Trova gli indici dei valori True nella riga
        if true_indices.numel() > false_values_for_row:  # Controlla se ci sono più di 15 valori True
            false_indices = torch.randperm(true_indices.numel())[
                            :true_indices.numel() - false_values_for_row]  # Seleziona casualmente gli indici da impostare a False
            mask[i, true_indices[false_indices]] = False


    x_ = torch.where(mask.to(x_.device), torch.tensor(0, dtype=x_.dtype).to(x_.device), x_)

    return x_
