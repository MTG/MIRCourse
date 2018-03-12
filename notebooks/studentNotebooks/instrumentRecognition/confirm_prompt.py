def confirm(prompt=None):    
    if prompt is None:
        prompt = 'Confirm'
    
    while True:
        ans = input(prompt)
        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False
