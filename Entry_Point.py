from tkinter import *
from tkinter import ttk
from HandPostureMain import MainFunction

final_dict = {}

root = ''
root1 = ''
preprocessing_var = ''
classfier_var = ''
dimensionality_reduction_var = ''
resultList = []

preprocessing_steps = ['Standarization', 'Normalization']

classifiers = ['All','Naive Bayes (Baseline)', 'Support Vector Machines (RBF Kernel)', 'Random Forest',
                   'K-Nearest Neighbors', 'Linear Discriminant Analysis',
                   'Quadratic Discriminant Analysis', 'Logistic Regression']

dr_techiniques = ['Principal Component Analysis', "Fischer's Linear Discriminant", 'Univariate Feature Selection',
                      'Mutual Info. Feature Selection']


def show_results():
    root1.destroy()
    initialize()


def new_window():
    global root1
    global resultList
    root1 = Tk()

    root1.title('MPR Project')
    myLabel = Label(root1, text="Motion Capture based Hand Posture Recognition", font='Helvetica 14 bold')
    myLabel.pack()

    mainframe = Frame(root1)
    # mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    mainframe.pack(pady=30, padx=80)

    ''' table '''
    Label(mainframe, text="Results", font='Helvetica 12 bold').grid(row=0, columnspan=3)

    cols = ('Classifier', 'Train Accuracy', 'Test Accuracy', 'F-Score')
    listBox = ttk.Treeview(mainframe, columns=cols, show='headings')

    for col in cols:
        listBox.heading(col, text=col)
    listBox.grid(row=1, column=0, columnspan=1)


    for i, (name, train_accuracy, test_accuracy, f_score) in enumerate(resultList, start=1):
        listBox.insert("", "end", values=(name, train_accuracy, test_accuracy, f_score))

    size = len(resultList)
    showScores = Button(mainframe, text="Main Menu", width=15, command=show_results)
    showScores.grid(row=size+1, column=0)

    root1.mainloop()

# CALL THE FUNCTION TO TRAIN MODEL - GET RESULTS and DISPLAY RESULTS
def submit_results():
    global root
    global resultList
    preprocessing_value = preprocessing_var.get()
    classifier_value = classfier_var.get()
    dr_value = dimensionality_reduction_var.get()

    p_index = preprocessing_steps.index(preprocessing_value)
    c_index = classifiers.index(classifier_value)
    d_index = dr_techiniques.index(dr_value)

    final_dict = {}
    final_dict["preprocessing"] = p_index
    final_dict["classifier"] = c_index
    final_dict["dimensionality_reduction"] = d_index

    #print(final_dict)
    root.destroy()

    ''' call the model, set result variables '''
    resultList = []
    resultList = MainFunction(final_dict)
    #resultList = [['Jim', '0.33', '1'], ['Dave', '0.67', '1'], ['James', '0.67', '1'], ['Eden', '0.5', '1']]
    new_window()



def initialize():
    global root
    global preprocessing_var
    global classfier_var
    global dimensionality_reduction_var


    root = Tk()
    root.title('MPR Project')
    myLabel = Label(root, text="Motion Capture based Hand Posture Recognition", font='Helvetica 18 bold')
    myLabel.pack()
    
    # grid layout
    mainframe = Frame(root)
    # mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    mainframe.columnconfigure(0, weight=1)
    mainframe.rowconfigure(0, weight=1)
    mainframe.pack(pady=30, padx=80)

    preprocessing_var = StringVar(root)
    classfier_var = StringVar(root)
    dimensionality_reduction_var = StringVar(root)

    preprocessing_var.set(preprocessing_steps[0])
    classfier_var.set(classifiers[0])
    dimensionality_reduction_var.set(dr_techiniques[0])

    preprocessing_menu = OptionMenu(mainframe, preprocessing_var, *preprocessing_steps)
    Label(mainframe, text="Preprocessing", font='Helvetica 12 bold').grid(row=1, column=1, padx=30, pady=20)
    preprocessing_menu.grid(row=2, column=1)

    dimensionality_reduction_menu = OptionMenu(mainframe, dimensionality_reduction_var, *dr_techiniques)
    Label(mainframe, text="Dimensionality Reduction Technique", font='Helvetica 12 bold').grid(row=1, column=2, padx=20,
                                                                                               pady=20)
    dimensionality_reduction_menu.grid(row=2, column=2)

    classifier_menu = OptionMenu(mainframe, classfier_var, *classifiers)
    Label(mainframe, text="Classifier", font='Helvetica 12 bold').grid(row=1, column=3, padx=30, pady=20)
    classifier_menu.grid(row=2, column=3)



    button = Button(mainframe, text="Show Results", command=submit_results)  # , highlightbackground='#3E4149')
    button.grid(row=3, column=4)
    

    ''' results '''

    root.mainloop()


if __name__ == "__main__":
    initialize()


