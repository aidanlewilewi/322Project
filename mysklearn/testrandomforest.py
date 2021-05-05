from mysklearn import myutils
from mysklearn.myclassifiers import MyRandomForest


def testRandomForest():


    interview_header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    interview_table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

    xTest = [
        ['Junior', 'Java', 'yes', 'no'],
        ['Junior', 'Java', 'yes', 'yes']
    ]
    expected = ['True', 'False']

    classes = []
    for row in interview_table:
        classes.append(row.pop(-1))


    randomForest = MyRandomForest()
    testSet = randomForest.fit(interview_table, classes, 50, 25, 4)
    
    predictions = randomForest.predict(xTest)

    assert predictions == randomForest.predict(xTest)
    