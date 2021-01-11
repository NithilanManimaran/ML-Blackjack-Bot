from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
import tensorflow
from keras.models import model_from_json
import numpy as np
import pandas as pd
from blackjack_sim import *


def model_decision(model, player_sum, has_ace, dealer_card_num, new_stack,
                   games_played, card_count):
    input_array = np.array([player_sum, 0, has_ace,
                            dealer_card_num]).reshape(1, -1)
    cc_array = pd.DataFrame.from_dict([card_count])
    input_array = np.concatenate([input_array, cc_array], axis=1)
    predict_correct = model.predict(input_array)
    if predict_correct >= 0.52:
        print('hit', predict_correct)
        return 1, predict_correct
    else:
        print('stay', predict_correct)
        return 0, predict_correct


def setup():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")

    card_count = {
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        'A': 0
    }
    return model, card_count

def newGameSetup():
    for key in card_count:
        card_count[key] = 0
    cardsLeft[0] = 312


def next_round():
    player_hand.clear()
    dealer_hand.clear()


def predict():
    player_sum = total_up(player_hand)
    has_ace = 'A' in player_hand
    dealer_card_num = dealer_hand[0]
    predicted_val = model_decision(model, player_sum, has_ace, dealer_card_num,
                                   0, 0, card_count)
    return predicted_val


def add_to_player_hand(card):
    player_hand.append(card)
    card_count[card] += 1
    cardsLeft[0] -= 1


def remove_from_player_hand():
    card = player_hand.pop()
    card_count[card] -= 1
    cardsLeft[0] += 1

def add_to_dealer_hand(card):
    dealer_hand.append(card)
    card_count[card] += 1
    cardsLeft[0] -= 1


def remove_from_dealer_hand():
    card = dealer_hand.pop()
    card_count[card] -= 1
    cardsLeft[0] += 1


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(838, 814)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName("centralwidget")
        self.playerA = QtWidgets.QPushButton(self.centralwidget)
        self.playerA.setGeometry(QtCore.QRect(50, 560, 61, 111))
        self.playerA.setObjectName("playerA")
        self.player2 = QtWidgets.QPushButton(self.centralwidget)
        self.player2.setGeometry(QtCore.QRect(120, 560, 61, 111))
        self.player2.setObjectName("player2")
        self.player3 = QtWidgets.QPushButton(self.centralwidget)
        self.player3.setGeometry(QtCore.QRect(190, 560, 61, 111))
        self.player3.setObjectName("player3")
        self.player4 = QtWidgets.QPushButton(self.centralwidget)
        self.player4.setGeometry(QtCore.QRect(260, 560, 61, 111))
        self.player4.setObjectName("player4")
        self.player5 = QtWidgets.QPushButton(self.centralwidget)
        self.player5.setGeometry(QtCore.QRect(330, 560, 61, 111))
        self.player5.setObjectName("player5")
        self.player6 = QtWidgets.QPushButton(self.centralwidget)
        self.player6.setGeometry(QtCore.QRect(400, 560, 61, 111))
        self.player6.setObjectName("player6")
        self.player7 = QtWidgets.QPushButton(self.centralwidget)
        self.player7.setGeometry(QtCore.QRect(470, 560, 61, 111))
        self.player7.setObjectName("player7")
        self.player8 = QtWidgets.QPushButton(self.centralwidget)
        self.player8.setGeometry(QtCore.QRect(540, 560, 61, 111))
        self.player8.setObjectName("player8")
        self.player9 = QtWidgets.QPushButton(self.centralwidget)
        self.player9.setGeometry(QtCore.QRect(610, 560, 61, 111))
        self.player9.setObjectName("player9")
        self.player10 = QtWidgets.QPushButton(self.centralwidget)
        self.player10.setGeometry(QtCore.QRect(680, 560, 61, 111))
        self.player10.setObjectName("player10")
        self.dealer2 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer2.setGeometry(QtCore.QRect(110, 10, 61, 111))
        self.dealer2.setObjectName("dealer2")
        self.dealer5 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer5.setGeometry(QtCore.QRect(320, 10, 61, 111))
        self.dealer5.setObjectName("dealer5")
        self.dealer10 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer10.setGeometry(QtCore.QRect(670, 10, 61, 111))
        self.dealer10.setObjectName("dealer10")
        self.dealer4 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer4.setEnabled(True)
        self.dealer4.setGeometry(QtCore.QRect(250, 10, 61, 111))
        self.dealer4.setObjectName("dealer4")
        self.dealer9 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer9.setGeometry(QtCore.QRect(600, 10, 61, 111))
        self.dealer9.setObjectName("dealer9")
        self.dealer6 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer6.setGeometry(QtCore.QRect(390, 10, 61, 111))
        self.dealer6.setObjectName("dealer6")
        self.dealer7 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer7.setGeometry(QtCore.QRect(460, 10, 61, 111))
        self.dealer7.setObjectName("dealer7")
        self.dealer8 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer8.setGeometry(QtCore.QRect(530, 10, 61, 111))
        self.dealer8.setObjectName("dealer8")
        self.dealerA = QtWidgets.QPushButton(self.centralwidget)
        self.dealerA.setGeometry(QtCore.QRect(40, 10, 61, 111))
        self.dealerA.setObjectName("dealerA")
        self.dealer3 = QtWidgets.QPushButton(self.centralwidget)
        self.dealer3.setGeometry(QtCore.QRect(180, 10, 61, 111))
        self.dealer3.setObjectName("dealer3")
        self.playerDelete = QtWidgets.QPushButton(self.centralwidget)
        self.playerDelete.setGeometry(QtCore.QRect(750, 560, 61, 111))
        self.playerDelete.setObjectName("playerDelete")
        self.dealerDelete = QtWidgets.QPushButton(self.centralwidget)
        self.dealerDelete.setGeometry(QtCore.QRect(740, 10, 61, 111))
        self.dealerDelete.setObjectName("dealerDelete")
        self.DealeHand = QtWidgets.QLabel(self.centralwidget)
        self.DealeHand.setGeometry(QtCore.QRect(370, 120, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.DealeHand.setFont(font)
        self.DealeHand.setObjectName("DealeHand")
        self.PlayerHand = QtWidgets.QLabel(self.centralwidget)
        self.PlayerHand.setGeometry(QtCore.QRect(380, 510, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.PlayerHand.setFont(font)
        self.PlayerHand.setObjectName("PlayerHand")
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(320, 700, 75, 23))
        self.predict.setObjectName("predict")
        self.nextHand = QtWidgets.QPushButton(self.centralwidget)
        self.nextHand.setGeometry(QtCore.QRect(470, 700, 75, 23))
        self.nextHand.setObjectName("nextHand")
        self.Prompt = QtWidgets.QLabel(self.centralwidget)
        self.Prompt.setGeometry(QtCore.QRect(260, 290, 400, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe MDL2 Assets")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.Prompt.setFont(font)
        self.Prompt.setObjectName("Prompt")
        self.CardLabel = QtWidgets.QLabel(self.centralwidget)
        self.CardLabel.setGeometry(QtCore.QRect(650, 270, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.CardLabel.setFont(font)
        self.CardLabel.setObjectName("CardLabel")
        self.NumberCardsLeft = QtWidgets.QLabel(self.centralwidget)
        self.NumberCardsLeft.setGeometry(QtCore.QRect(720, 300, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.NumberCardsLeft.setFont(font)
        self.NumberCardsLeft.setObjectName("NumberCardsLeft")
        self.newGame = QtWidgets.QPushButton(self.centralwidget)
        self.newGame.setGeometry(QtCore.QRect(740, 700, 75, 23))
        self.newGame.setObjectName("newGame")
        self.PlayerCards = QtWidgets.QLabel(self.centralwidget)
        self.PlayerCards.setGeometry(QtCore.QRect(290, 400, 411, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe MDL2 Assets")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.PlayerCards.setFont(font)
        self.PlayerCards.setObjectName("PlayerCards")
        self.DealerCards = QtWidgets.QLabel(self.centralwidget)
        self.DealerCards.setGeometry(QtCore.QRect(280, 190, 411, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe MDL2 Assets")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.DealerCards.setFont(font)
        self.DealerCards.setObjectName("DealerCards")
        self.NextWarning = QtWidgets.QLabel(self.centralwidget)
        self.NextWarning.setGeometry(QtCore.QRect(220, 740, 481, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe MDL2 Assets")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.NextWarning.setFont(font)
        self.NextWarning.setObjectName("NextWarning")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 838, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)


        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.playerA.clicked.connect(self.playerAClicked)
        self.player2.clicked.connect(self.player2Clicked)
        self.player3.clicked.connect(self.player3Clicked)
        self.player4.clicked.connect(self.player4Clicked)
        self.player5.clicked.connect(self.player5Clicked)
        self.player6.clicked.connect(self.player6Clicked)
        self.player7.clicked.connect(self.player7Clicked)
        self.player8.clicked.connect(self.player8Clicked)
        self.player9.clicked.connect(self.player9Clicked)
        self.player10.clicked.connect(self.player10Clicked)
        self.playerDelete.clicked.connect(self.playerDeleteClicked)
        self.predict.clicked.connect(self.predictClicked)
        self.dealerA.clicked.connect(self.dealerAClicked)
        self.dealer2.clicked.connect(self.dealer2Clicked)
        self.dealer3.clicked.connect(self.dealer3Clicked)
        self.dealer4.clicked.connect(self.dealer4Clicked)
        self.dealer5.clicked.connect(self.dealer5Clicked)
        self.dealer6.clicked.connect(self.dealer6Clicked)
        self.dealer7.clicked.connect(self.dealer7Clicked)
        self.dealer8.clicked.connect(self.dealer8Clicked)
        self.dealer9.clicked.connect(self.dealer9Clicked)
        self.dealer10.clicked.connect(self.dealer10Clicked)
        self.dealerDelete.clicked.connect(self.dealerDeleteClicked)
        self.nextHand.clicked.connect(self.nextHandClicked)
        self.newGame.clicked.connect(self.newGameClicked)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.playerA.setText(_translate("MainWindow", "Ace"))
        self.player2.setText(_translate("MainWindow", "2"))
        self.player3.setText(_translate("MainWindow", "3"))
        self.player4.setText(_translate("MainWindow", "4"))
        self.player5.setText(_translate("MainWindow", "5"))
        self.player6.setText(_translate("MainWindow", "6"))
        self.player7.setText(_translate("MainWindow", "7"))
        self.player8.setText(_translate("MainWindow", "8"))
        self.player9.setText(_translate("MainWindow", "9"))
        self.player10.setText(_translate("MainWindow", "10"))
        self.dealer2.setText(_translate("MainWindow", "2"))
        self.dealer5.setText(_translate("MainWindow", "5"))
        self.dealer10.setText(_translate("MainWindow", "10"))
        self.dealer4.setText(_translate("MainWindow", "4"))
        self.dealer9.setText(_translate("MainWindow", "9"))
        self.dealer6.setText(_translate("MainWindow", "6"))
        self.dealer7.setText(_translate("MainWindow", "7"))
        self.dealer8.setText(_translate("MainWindow", "8"))
        self.dealerA.setText(_translate("MainWindow", "A"))
        self.dealer3.setText(_translate("MainWindow", "3"))
        self.playerDelete.setText(_translate("MainWindow", "Delete"))
        self.dealerDelete.setText(_translate("MainWindow", "Delete"))
        self.DealeHand.setText(_translate("MainWindow", "Dealer Hand"))
        self.PlayerHand.setText(_translate("MainWindow", "Player Hand"))
        self.predict.setText(_translate("MainWindow", "Predict"))
        self.nextHand.setText(_translate("MainWindow", "Next Hand"))
        self.Prompt.setText(_translate("MainWindow", "PROMPT"))
        self.CardLabel.setText(_translate("MainWindow", "Cards Left 6 decks"))
        self.NumberCardsLeft.setText(_translate("MainWindow", "312"))
        self.newGame.setText(_translate("MainWindow", "New Game"))
        self.PlayerCards.setText(_translate("MainWindow", "Show Player Cards"))
        self.DealerCards.setText(_translate("MainWindow", "Show Dealer Cards"))
        self.NextWarning.setText(_translate("MainWindow", "INPUT ALL SHOWN CARDS BEFORE NEXT HAND"))


    def playerAClicked(self):
        add_to_player_hand("A")
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player2Clicked(self):
        add_to_player_hand(2)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player3Clicked(self):
        add_to_player_hand(3)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player4Clicked(self):
        add_to_player_hand(4)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player5Clicked(self):
        add_to_player_hand(5)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player6Clicked(self):
        add_to_player_hand(6)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player7Clicked(self):
        add_to_player_hand(7)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player8Clicked(self):
        add_to_player_hand(8)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player9Clicked(self):
        add_to_player_hand(9)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def player10Clicked(self):
        add_to_player_hand(10)
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def playerDeleteClicked(self):
        if len(player_hand) == 0:
            return
        remove_from_player_hand()
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def predictClicked(self):
        if len(dealer_hand) != 1:
            self.Prompt.setText("Dealer MUST show 1 card")
        else:
            prediction = predict()
            if(prediction[0] == 1):
                self.Prompt.setText("Hit[" + str(prediction[1][0][0]) + "]")
            else:
                self.Prompt.setText("Stay[" + str(prediction[1][0][0]) + "]")

    def dealerAClicked(self):
        add_to_dealer_hand("A")
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer2Clicked(self):
        add_to_dealer_hand(2)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer3Clicked(self):
        add_to_dealer_hand(3)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer4Clicked(self):
        add_to_dealer_hand(4)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer5Clicked(self):
        add_to_dealer_hand(5)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer6Clicked(self):
        add_to_dealer_hand(6)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer7Clicked(self):
        add_to_dealer_hand(7)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer8Clicked(self):
        add_to_dealer_hand(8)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer9Clicked(self):
        add_to_dealer_hand(9)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealer10Clicked(self):
        add_to_dealer_hand(10)
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def dealerDeleteClicked(self):
        if len(dealer_hand) == 0:
            return
        remove_from_dealer_hand()
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.NumberCardsLeft.setText(str(cardsLeft))

    def nextHandClicked(self):
        next_round()
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.Prompt.setText("All cards added, new round begins")

    def newGameClicked(self):
        newGameSetup()
        self.DealerCards.setText("Dealer Hand: " + dealer_hand.__str__())
        self.PlayerCards.setText("Player Hand: " + player_hand.__str__())
        self.Prompt.setText("A new game begins")
        self.NumberCardsLeft.setText(str(cardsLeft))


if __name__ == "__main__":
    import sys

    player_hand = []
    dealer_hand = []
    model, card_count = setup()

    cardsLeft = [312]

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
