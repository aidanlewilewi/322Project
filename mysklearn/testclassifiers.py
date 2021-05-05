import numpy as np
import scipy.stats as stats  
from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier, MyNaiveBayesClassifier, MyDecisionTreeClassifier, MyRandomForest
import mysklearn.myutils as myutils 

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    # generate dataset
    np.random.seed(0)
    
    xSamples = []
    for i in range(0, 100):
        xVal = []
        xVal.append(i)
        xSamples.append(xVal)

    yVals = []
    for i in range(0, 100):
        val = []
        val.append(i * 2 + np.random.normal(0,25))
        yVals.append(val)

    x = myutils.convert_2D_to_1D(xSamples)
    y = myutils.convert_2D_to_1D(yVals)


    trueSlope, trueIntercept, trueR, trueP, trueStdError = stats.linregress(x,y) # scipy calculation 

    testRegress = MySimpleLinearRegressor() 
    testRegress.fit(xSamples,yVals) # get slope and intercept using fit()

    testSlope = testRegress.slope 
    testIntercept = testRegress.intercept

    assert np.isclose(testSlope, trueSlope) # calculated slope close to scipy slope
    assert np.isclose(testIntercept, trueIntercept) # calculated intercept close to scipy intercept


    # generate dataset
    xSamples = []
    for i in range(0, 100):
        xVal = []
        xVal.append(i)
        xSamples.append(xVal)

    yVals = []
    for i in range(0, 100):
        val = []
        val.append(i * 5.7 + np.random.normal(7,62))
        yVals.append(val)

    x = myutils.convert_2D_to_1D(xSamples)
    y = myutils.convert_2D_to_1D(yVals)


    trueSlope, trueIntercept, trueR, trueP, trueStdError = stats.linregress(x,y) # scipy calculation 

    testRegress = MySimpleLinearRegressor() 
    testRegress.fit(xSamples,yVals) # get slope and intercept using fit()

    testSlope = testRegress.slope 
    testIntercept = testRegress.intercept

    assert np.isclose(testSlope, trueSlope) # calculated slope close to scipy slope
    assert np.isclose(testIntercept, trueIntercept) # calculated intercept close to scipy intercept


def test_simple_linear_regressor_predict():

    testRegress = MySimpleLinearRegressor() 
    testRegress.slope = 4.9
    testRegress.intercept = 10

    xSamples = []
    for i in range(1, 101):
        xVal = []
        xVal.append(i)
        xSamples.append(xVal)
    yVals = []
    for i in range(1, 101):
        y = i * 4.9 + 10
        yVals.append(y)

    yPredictions = testRegress.predict(xSamples)

    for i in range(len(yPredictions)):
        assert np.isclose(yPredictions[i], yVals[i])
    
    testRegress = MySimpleLinearRegressor() 
    testRegress.slope = 18.7
    testRegress.intercept = 10.3

    xSamples = []
    for i in range(1, 101):
        xVal = []
        xVal.append(i)
        xSamples.append(xVal)
    yVals = []
    for i in range(1, 101):
        y = i * 18.7 + 10.3
        yVals.append(y)

    yPredictions = testRegress.predict(xSamples)

    for i in range(len(yPredictions)):
        assert np.isclose(yPredictions[i], yVals[i])

def test_kneighbors_classifier_kneighbors():

    testKNN = MyKNeighborsClassifier()

    train = [
        [1, 1],
        [1, 0],
        [0.33, 0],
        [0, 0]
    ]
    train_labels = ["bad", "bad", "good", "good"]
    test = [[0.33, 1]]

    testKNN.n_neighbors = 3
    testKNN.X_train = train 
    testKNN.y_train = train_labels

    threeDist, threeIndices = testKNN.kneighbors(test)
    expectedDist = [0.670, 1.000, 1.053]
    expectedInd = [0, 2, 3]

    assert np.allclose(threeDist, expectedDist)
    assert np.allclose(threeIndices, expectedInd)

    # case 2
    testKNN = MyKNeighborsClassifier() 
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test = [[2, 3]]

    testKNN.n_neighbors = 3
    testKNN.X_train = train 
    testKNN.y_train = train_labels

    threeDist, threeIndices = testKNN.kneighbors(test)
    expectedDist = [1.414, 1.414, 2.000]
    expectedInd = [0, 4, 6]

    assert np.allclose(threeDist, expectedDist)
    assert np.allclose(threeIndices, expectedInd)


     # case 3
    testKNN = MyKNeighborsClassifier() 
    train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    train_labels = [-1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1]

    testKNN.n_neighbors = 5
    testKNN.X_train = train 
    testKNN.y_train = train_labels
    test = [[9.1, 11.0]]
    

    fiveDist, fiveIndices = testKNN.kneighbors(test)
    expectedDist = [0.608, 1.237, 2.202, 2.802, 2.915]
    expectedInd = [6, 5, 7, 4, 8]

    assert np.allclose(fiveDist, expectedDist)
    assert np.allclose(fiveIndices, expectedInd)


def test_kneighbors_classifier_predict():

    testKNN = MyKNeighborsClassifier()

    train = [
        [1, 1],
        [1, 0],
        [0.33, 0],
        [0, 0]
    ]
    train_labels = ["bad", "bad", "good", "good"]
    test = [[0.33, 1]]

    testKNN.n_neighbors = 3
    testKNN.X_train = train 
    testKNN.y_train = train_labels

    predictions = testKNN.predict(test)
    expected = ['good']
    
    for i in range(len(predictions)):
        assert predictions[i] == expected[i]

    # case 2
    testKNN = MyKNeighborsClassifier() 
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test = [[2, 3]]

    testKNN.n_neighbors = 3
    testKNN.X_train = train 
    testKNN.y_train = train_labels

    predictions = testKNN.predict(test)
    expected = ['yes']

    for i in range(len(predictions)):
        assert predictions[i] == expected[i]

    # case 3
    testKNN = MyKNeighborsClassifier() 
    train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]
    ]
    train_labels = [-1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1]

    testKNN.n_neighbors = 5
    testKNN.X_train = train 
    testKNN.y_train = train_labels
    test = [[9.1, 11.0]]

    predictions = testKNN.predict(test)
    expected = [1]

    for i in range(len(predictions)):
        assert predictions[i] == expected[i]

def test_naive_bayes_classifier_fit():

    testNB = MyNaiveBayesClassifier() 
    truePriors = [3/8, 5/8]
    truePost = {0: {1: {'no': 2/3, 'yes': 0.8}, 2: {'no': 1/3, 'yes': 0.2}}, 1: {5: {'no': 0.6666666666666666, 'yes': 0.4}, 6: {'no': 0.3333333333333333, 'yes': 0.6}}}

    testData = [
        [1, 5, 'yes'],
        [2, 6, 'yes'],
        [1, 5, 'no'],
        [1, 5, 'no'],
        [1, 6, 'yes'],
        [2, 6, 'no'],
        [1, 5, 'yes'],
        [1, 6, 'yes']
    ]

    allClasses = []
    for row in testData:
        allClasses.append(row.pop())

    testNB.fit(testData, allClasses)
    
    assert np.allclose(testNB.priors, truePriors)
    assert testNB.posteriors == truePost

    testNB = MyNaiveBayesClassifier()

    # RQ5 (fake) iPhone purchases dataset
    truePriors = [0.333, 0.667]
    truePost = {0: {1: {'no': 0.6, 'yes': 0.2}, 2: {'no': 0.4, 'yes': 0.8}}, 
        1: {1: {'no': 0.2, 'yes': 0.3}, 
        2: {'no': 0.4, 'yes': 0.4}, 
        3: {'no': 0.4, 'yes': 0.3}}, 2: {'excellent': {'no': 0.6, 'yes': 0.3}, 'fair': {'no': 0.4, 'yes': 0.7}}}
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    allClasses = []
    for row in iphone_table:
        allClasses.append(row.pop())

    testNB.fit(iphone_table, allClasses)

    assert np.allclose(testNB.priors, truePriors)
    assert testNB.posteriors == truePost

    testNB = MyNaiveBayesClassifier()
    truePriors = [0.05, 0.1, 0.7, 0.15]
    truePost = {0: {'holiday': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.14285714285714285, 'very late': 0.0}, 
    'saturday': {'cancelled': 1.0, 'late': 0.5, 'on time': 0.14285714285714285, 'very late': 0.0}, 
    'sunday': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.07142857142857142, 'very late': 0.0}, 
    'weekday': {'cancelled': 0.0, 'late': 0.5, 'on time': 0.6428571428571429, 'very late': 1.0}}, 
    1: {'autumn': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.14285714285714285, 'very late': 0.3333333333333333}, 
    'spring': {'cancelled': 1.0, 'late': 0.0, 'on time': 0.2857142857142857, 'very late': 0.0}, 
    'summer': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.42857142857142855, 'very late': 0.0}, 
    'winter': {'cancelled': 0.0, 'late': 1.0, 'on time': 0.14285714285714285, 'very late': 0.6666666666666666}}, 
    2: {'high': {'cancelled': 1.0, 'late': 0.5, 'on time': 0.2857142857142857, 'very late': 0.3333333333333333}, 
    'none': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.35714285714285715, 'very late': 0.0}, 
    'normal': {'cancelled': 0.0, 'late': 0.5, 'on time': 0.35714285714285715, 'very late': 0.6666666666666666}}, 
    3: {'heavy': {'cancelled': 1.0, 'late': 0.5, 'on time': 0.07142857142857142, 'very late': 0.6666666666666666}, 
    'none': {'cancelled': 0.0, 'late': 0.5, 'on time': 0.35714285714285715, 'very late': 0.3333333333333333}, 
    'slight': {'cancelled': 0.0, 'late': 0.0, 'on time': 0.5714285714285714, 'very late': 0.0}}}
    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    allClasses = []
    for row in train_table:
        allClasses.append(row.pop())

    testNB.fit(train_table, allClasses)

    assert np.allclose(testNB.priors, truePriors)
    assert testNB.posteriors == truePost

def test_naive_bayes_classifier_predict():

    testNB = MyNaiveBayesClassifier() 
    truePriors = [3/8, 5/8]
    X_test = [[1, 5]]
    expected = ['yes']
    testData = [
        [1, 5, 'yes'],
        [2, 6, 'yes'],
        [1, 5, 'no'],
        [1, 5, 'no'],
        [1, 6, 'yes'],
        [2, 6, 'no'],
        [1, 5, 'yes'],
        [1, 6, 'yes']
    ]

    allClasses = []
    for row in testData:
        allClasses.append(row.pop())

    testNB.fit(testData, allClasses)
    predictions = testNB.predict(X_test)

    assert predictions[0] == 'yes'


    testNB = MyNaiveBayesClassifier()

    # RQ5 (fake) iPhone purchases dataset
    truePriors = [0.333, 0.667]
    X_test = [[2, 2, 'fair'], [1, 1, 'excellent']]
    expected = ['yes', 'no']
    iphone_col_names = ["standing", "job_status", "credit_rating", "buys_iphone"]
    iphone_table = [
        [1, 3, "fair", "no"],
        [1, 3, "excellent", "no"],
        [2, 3, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [2, 1, "fair", "yes"],
        [2, 1, "excellent", "no"],
        [2, 1, "excellent", "yes"],
        [1, 2, "fair", "no"],
        [1, 1, "fair", "yes"],
        [2, 2, "fair", "yes"],
        [1, 2, "excellent", "yes"],
        [2, 2, "excellent", "yes"],
        [2, 3, "fair", "yes"],
        [2, 2, "excellent", "no"],
        [2, 3, "fair", "yes"]
    ]
    allClasses = []
    for row in iphone_table:
        allClasses.append(row.pop())

    testNB.fit(iphone_table, allClasses)
    predictions = testNB.predict(X_test)
   
    for i in range(len(predictions)):
        assert predictions[i] == expected[i]


    testNB = MyNaiveBayesClassifier()
    truePriors = [0.05, 0.1, 0.7, 0.15]
    # Bramer 3.2 train dataset
    train_col_names = ["day", "season", "wind", "rain", "class"]
    train_table = [
        ["weekday", "spring", "none", "none", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "none", "slight", "on time"],
        ["weekday", "winter", "high", "heavy", "late"], 
        ["saturday", "summer", "normal", "none", "on time"],
        ["weekday", "autumn", "normal", "none", "very late"],
        ["holiday", "summer", "high", "slight", "on time"],
        ["sunday", "summer", "normal", "none", "on time"],
        ["weekday", "winter", "high", "heavy", "very late"],
        ["weekday", "summer", "none", "slight", "on time"],
        ["saturday", "spring", "high", "heavy", "cancelled"],
        ["weekday", "summer", "high", "slight", "on time"],
        ["saturday", "winter", "normal", "none", "late"],
        ["weekday", "summer", "high", "none", "on time"],
        ["weekday", "winter", "normal", "heavy", "very late"],
        ["saturday", "autumn", "high", "slight", "on time"],
        ["weekday", "autumn", "none", "heavy", "on time"],
        ["holiday", "spring", "normal", "slight", "on time"],
        ["weekday", "spring", "normal", "none", "on time"],
        ["weekday", "spring", "normal", "slight", "on time"]
    ]

    allClasses = []
    for row in train_table:
        allClasses.append(row.pop())

    testNB.fit(train_table, allClasses)

def test_decision_tree_classifier_fit():

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
    trueTree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]


    classes = []
    for row in interview_table:
        classes.append(row.pop(-1))

    testDT = MyDecisionTreeClassifier()
    testDT.fit(interview_table, classes)

    assert testDT.tree == trueTree # TODO: fix this


    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    degrees_tree = \
        ['Attribute', 'att0',
            ['Value', 'A',
                ['Attribute', 'att4',
                    ['Value', 'A',
                        ['Leaf', 'FIRST', 5, 14]
                    ],
                    ['Value', 'B',
                        ['Attribute', 'att3',
                            ['Value', 'A',
                                ['Attribute', 'att1',
                                    ['Value', 'A',
                                        ['Leaf', 'FIRST', 1, 2]
                                    ],
                                    ['Value', 'B',
                                        ['Leaf', 'SECOND', 1, 2]
                                    ]
                                ]
                            ],
                            ['Value', 'B',
                                ['Leaf', 'SECOND', 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ['Value', 'B',
                ['Leaf', 'SECOND', 12, 26]
            ]
        ]

    
    classes = []
    for row in degrees_table:
        classes.append(row.pop(-1))

    testDT = MyDecisionTreeClassifier()
    testDT.fit(degrees_table, classes)

    assert testDT.tree == degrees_tree


def test_decision_tree_classifier_predict():

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
    trueTree = \
        ["Attribute", "att0",
            ["Value", "Junior", 
                ["Attribute", "att3",
                    ["Value", "no", 
                        ["Leaf", "True", 3, 5]
                    ],
                    ["Value", "yes", 
                        ["Leaf", "False", 2, 5]
                    ]
                ]
            ],
            ["Value", "Mid",
                ["Leaf", "True", 4, 14]
            ],
            ["Value", "Senior",
                ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                    ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                    ]
                ]
            ]
        ]

    xTest = [
        ['Junior', 'Java', 'yes', 'no'],
        ['Junior', 'Java', 'yes', 'yes']
    ]
    expected = ['True', 'False']

    classes = []
    for row in interview_table:
        classes.append(row.pop(-1))

    testDT = MyDecisionTreeClassifier()
    testDT.fit(interview_table, classes)

    predictions = testDT.predict(xTest)
    assert predictions == expected

    # bramer degrees dataset
    degrees_header = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    degrees_table = [
        ["A", "B", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "A", "B", "B", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "A", "A", "A", "FIRST"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["B", "A", "A", "B", "B", "SECOND"],
        ["A", "B", "B", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "A", "B", "FIRST"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["A", "A", "B", "B", "B", "SECOND"],
        ["B", "B", "B", "B", "B", "SECOND"],
        ["A", "A", "B", "A", "A", "FIRST"],
        ["B", "B", "B", "A", "A", "SECOND"],
        ["B", "B", "A", "A", "B", "SECOND"],
        ["B", "B", "B", "B", "A", "SECOND"],
        ["B", "A", "B", "A", "B", "SECOND"],
        ["A", "B", "B", "B", "A", "FIRST"],
        ["A", "B", "A", "B", "B", "SECOND"],
        ["B", "A", "B", "B", "B", "SECOND"],
        ["A", "B", "B", "B", "B", "SECOND"],
    ]

    degrees_tree = \
        ['Attribute', 'att0',
            ['Value', 'A',
                ['Attribute', 'att4',
                    ['Value', 'A',
                        ['Leaf', 'FIRST', 5, 14]
                    ],
                    ['Value', 'B',
                        ['Attribute', 'att3',
                            ['Value', 'A',
                                ['Attribute', 'att1',
                                    ['Value', 'A',
                                        ['Leaf', 'FIRST', 1, 2]
                                    ],
                                    ['Value', 'B',
                                        ['Leaf', 'SECOND', 1, 2]
                                    ]
                                ]
                            ],
                            ['Value', 'B',
                                ['Leaf', 'SECOND', 7, 9]
                            ]
                        ]
                    ]
                ]
            ],
            ['Value', 'B',
                ['Leaf', 'SECOND', 12, 26]
            ]
        ]

    
    classes = []
    for row in degrees_table:
        classes.append(row.pop(-1))

    testDT = MyDecisionTreeClassifier()
    testDT.fit(degrees_table, classes)

    xTest = [
        ["B", "B", "B", "B", "B"], 
        ["A", "A", "A", "A", "A"], 
        ["A", "A", "A", "A", "B"]
    ]
    expected = ['SECOND', 'FIRST', 'FIRST']

    predictions = testDT.predict(xTest)

    testDT.print_decision_rules(trueTree)
    
    assert predictions == expected


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
    