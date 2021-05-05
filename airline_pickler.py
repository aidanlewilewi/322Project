import pickle 

airline_header = ["att0", "att1", "att2", "att3", "att4","att5"]

airline_tree =  \
['Attribute', 'att4', 
    ['Value', 'Business', 
        ['Attribute', 'att3', 
            ['Value', 'Business travel', 
                ['Attribute', 'att1', 
                    ['Value', 'Loyal Customer', 
                        ['Attribute', 'att2', 
                            ['Value', 1, 
                                ['Leaf', 'neutral or dissatisfied', 11, 16]
                            ], 
                            ['Value', 2, 
                                ['Attribute', 'att5', 
                                    ['Value', 1, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 12, 20]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 13, 20]
                                            ]
                                        ]
                                    ],
                                    ['Value', 2, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 11, 19]
                                            ],
                                            ['Value', 'Male', 
                                                ['Leaf', 'neutral or dissatisfied', 7, 13]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 3, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'neutral or dissatisfied', 7, 13]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'neutral or dissatisfied', 4, 6]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 4, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 10, 15]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 9, 11]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 5, 
                                        ['Leaf', 'satisfied', 1, 118]
                                    ]
                                ]
                            ], 
                            ['Value', 3, 
                                ['Attribute', 'att5', 
                                    ['Value', 1, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 19, 25]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 17, 21]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 2, 
                                        ['Attribute', 'att0',  
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 18, 23]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 20, 22]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 3, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female', 
                                                ['Leaf', 'satisfied', 17, 25]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 17, 20]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 4, 
                                        ['Attribute', 'att0', 
                                            ['Value', 'Female',    
                                                ['Leaf', 'satisfied', 15, 17]
                                            ], 
                                            ['Value', 'Male', 
                                                ['Leaf', 'satisfied', 16, 20]
                                            ]
                                        ]
                                    ], 
                                    ['Value', 5,    
                                        ['Leaf', 'satisfied', 2, 175]
                                    ]
                                ]
                            ], 
                            ['Value', 4, 
                                ['Leaf', 'satisfied', 59, 74]
                            ],
                            ['Value', 5, 
                                ['Leaf', 'satisfied', 4, 5]
                            ]
                        ]
                    ], 
                    ['Value', 'disloyal Customer', 
                        ['Attribute', 'att2', 
                            ['Value', 1, 
                                ['Attribute', 'att0', 
                                    ['Value', 'Female', 
                                        ['Leaf', 'satisfied', 2, 5]
                                    ], 
                                    ['Value', 'Male', 
                                        ['Leaf', 'neutral or dissatisfied', 2, 3]
                                    ]
                                ]
                            ], 
                            ['Value', 2, 
                                ['Leaf', 'neutral or dissatisfied', 28, 37]
                            ],
                            ['Value', 3, 
                                ['Leaf', 'neutral or dissatisfied', 13, 18]
                            ], 
                            ['Value', 4, 
                                ['Attribute', 'att0', 
                                    ['Value', 'Female', 
                                        ['Leaf', 'neutral or dissatisfied', 2, 3]
                                    ],
                                    ['Value', 'Male', 
                                        ['Leaf', 'neutral or dissatisfied', 2, 5]
                                    ]
                                ]
                            ], 
                            ['Value', 5, 
                                ['Leaf', 'neutral or dissatisfied', 1, 2]
                            ]
                        ]
                    ]
                ]
            ],
            ['Value', 'Personal Travel', 
                ['Leaf', 'neutral or dissatisfied', 22, 23]
            ]
        ]
    ], 
    ['Value', 'Eco', 
        ['Attribute', 'att3', 
            ['Value', 'Business travel', 
                ['Attribute', 'att1', 
                    ['Value', 'Loyal Customer', 
                        ['Leaf', 'neutral or dissatisfied', 63, 90]
                    ],
                    ['Value', 'disloyal Customer', 
                        ['Attribute', 'att0', 
                            ['Value', 'Female', 
                                ['Leaf', 'neutral or dissatisfied', 45, 52]
                            ], 
                            ['Value', 'Male', 
                                ['Leaf', 'neutral or dissatisfied', 36, 88]
                            ]
                        ]
                    ]
                ]
            ], 
            ['Value', 'Personal Travel', 
                ['Attribute', 'att2', 
                    ['Value', 1, 
                        ['Leaf', 'neutral or dissatisfied', 64, 66]
                    ], 
                    ['Value', 2, 
                        ['Leaf', 'neutral or dissatisfied', 55, 57]
                    ], 
                    ['Value', 3, 
                        ['Leaf', 'neutral or dissatisfied', 57, 61]
                    ], 
                    ['Value', 4, 
                        ['Leaf', 'neutral or dissatisfied', 55, 242]
                    ],
                    ['Value', 5, 
                        ['Leaf', 'neutral or dissatisfied', 3, 242]
                    ]
                ]
            ]
        ]
    ], 
    ['Value', 'Eco Plus', 
        ['Attribute', 'att2', 
            ['Value', 1, 
                ['Attribute', 'att3',  
                    ['Value', 'Business travel', 
                        ['Attribute', 'att0', 
                            ['Value', 'Female', 
                                ['Leaf', 'neutral or dissatisfied', 6, 10]
                            ], 
                            ['Value', 'Male', 
                                ['Leaf', 'neutral or dissatisfied', 3, 4]
                            ]
                        ]
                    ], 
                    ['Value', 'Personal Travel', 
                        ['Leaf', 'neutral or dissatisfied', 11, 21]
                    ]
                ]
            ], 
            ['Value', 2, 
                ['Leaf', 'neutral or dissatisfied', 32, 36]
            ], 
            ['Value', 3, 
                ['Leaf', 'neutral or dissatisfied', 14, 20]
            ], 
            ['Value', 4, 
                ['Leaf', 'neutral or dissatisfied', 17, 19]
            ], 
            ['Value', 5, 
                ['Leaf', 'neutral or dissatisfied', 1, 97]
            ]
        ]
    ]
]

packaged_object = [airline_header, airline_tree]
outfile = open("tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()


# deserialize to object (unpickle)
infile = open("tree.p", "rb")
header2, airline_tree2 = pickle.load(infile)
infile.close()

# check
print(header2)
print(airline_tree2)