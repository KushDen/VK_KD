
def flatten(l):
    return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    mystr2 = [x for x in mystr if x not in tokens_to_remove]
    if (mystr2.count('.') > 0): mystr2 = mystr2[:mystr2.index('.')] # удаляю все символы после точки
    return mystr2


def get_text(x, TRG_vocab):
    text = remove_tech_tokens([TRG_vocab.itos[token] for token in x])
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()
    output = output[1:] # удаляю первое предсказанное слово
    original = [TRG_vocab.itos[x] for x in list(trg[:,0].cpu().numpy())]
    generated = [TRG_vocab.itos[x] for x in list(output[:, 0])]
    
    original = remove_tech_tokens(original)
    generated = remove_tech_tokens(generated)
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()
