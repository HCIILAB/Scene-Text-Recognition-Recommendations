import string
def get_alphabets(type):
    if type == 'casia_360cc':
        with open('lib/datasets/mix_cc_casia.txt','rb') as f:
            res = [char.strip().decode('gbk','ignore') for char in f.readlines()]
    elif type == 'lowercase':
        res = string.digits + string.ascii_lowercase
    elif type == 'allcases':
        res = string.digits + string.ascii_letters
    elif type == 'allcases_symbols':
        res = string.printable[:-6]
    else:
        assert True, 'Wrong alphabets'
    print(res)
    return res