import csv

pathcommenttrain = './sc-data/train_comment_clean.csv'
pathfunctiontrain = './sc-data/train_function_clean.csv'
pathcommenttest = './sc-data/test_comment_clean.csv'
pathfunctiontest = './sc-data/test_function_clean.csv'

savepathcommenttrain = './data/comment_train.csv'
savepathfunctiontrain = './data/function_train.csv'
savepathcommenttest = './data/comment_test.csv'
savepathfunctiontest = './data/function_test.csv'

def get_sig(func):
    '''get signature from a function string'''
    return func[:func.find('{')]


if __name__ == '__main__':
    comments = []
    sigs = []
    funcs = []
    with open(pathcommenttrain, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            comment = '/*' + row[0] + '*/'
            comments.append(comment)
    with open(pathfunctiontrain, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for index, row in enumerate(rows):
            func = row[0]
            funcs.append(func)
            comments[index] = comments[index] + ' ' + get_sig(func)
    with open(savepathcommenttrain, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')
        for row in comments:
            spamwriter.writerow([row])
    with open(savepathfunctiontrain, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')
        for row in funcs:
            spamwriter.writerow([row])

    with open(pathcommenttest, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            comment = '/*' + row[0] + '*/'
            comments.append(comment)
    with open(pathfunctiontest, newline='') as csvfile:
        rows = csv.reader(csvfile)
        for index, row in enumerate(rows):
            func = row[0]
            funcs.append(func)
            comments[index] = comments[index] + ' ' + get_sig(func)
    with open(savepathcommenttest, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')
        for row in comments:
            spamwriter.writerow([row])
    with open(savepathfunctiontest, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')
        for row in funcs:
            spamwriter.writerow([row])
