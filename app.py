import cv2
import face_recognition
import os
import pickle
from tinydb import TinyDB, Query
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QThread, pyqtSignal


known_persons = pickle.loads(
    open(
        os.path.join('./encoded_dataset.pkl'),
        'rb'
    ).read()
)

names = [name for name in known_persons]
activities = ['entered room', 'left room']

class VideoThread(QThread):

    prepared_data = pyqtSignal(dict)

    previous_matches = []

    def run(self):
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:

            ret, frame = cap.read()

            data = {
                'frame': None,
                'matches': [],
                'unknown': []
            }

            if ret:

                faces = face_recognition.face_locations(frame, model='cnn')
                encoded_faces = face_recognition.face_encodings(frame, faces)
                matches, unknown = self.find_matches(encoded_faces, faces)

                for match in matches:

                    name, face = list(match.items())[0]
                    self.draw_box_with_label(name, face, frame)
                    if name not in self.previous_matches:
                        data['matches'].append(name)

                for face in unknown:

                    self.draw_box_with_label(None, face, frame)
                    data['unknown'].append(True)

                self.previous_matches = list(map(
                    lambda element: list(element.items())[0][0],
                    matches
                ))

                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                ConvertToQtFormat = QImage(
                    bgr_frame.data,
                    bgr_frame.shape[1],
                    bgr_frame.shape[0],
                    QImage.Format_RGB888
                )

                scaled_frame = ConvertToQtFormat.scaled(
                    1253,
                    720,
                    QtCore.Qt.KeepAspectRatio
                )

                data['frame'] = scaled_frame
                self.prepared_data.emit(data)


    def find_matches(self, encoded_faces, faces):
        global known_persons

        matches = []
        unknown = []

        for encoded_face, face in zip(encoded_faces, faces):

            didMatch = False

            for person in known_persons:

                candidates = face_recognition.compare_faces(
                    known_persons[person],
                    encoded_face
                )

                if True in candidates:

                    didMatch = True
                    matches.append({ person: face })

            if not didMatch:

                unknown.append(face)

        return matches, unknown


    def draw_box_with_label(self, name, img, frame):

        color = (0, 255, 0) if name else (0, 0, 255)
        name = name if name else "Unknown"
        top_left = (img[3], img[0])
        bot_right = (img[1], img[2])

        cv2.rectangle(
            frame,
            top_left,
            bot_right,
            color,
            1
        )

        text_top_left = (img[3], img[2])
        text_bot_right = (img[1], img[2] + 20)

        text_size = cv2.getTextSize(
            name,
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            2
        )

        text_offset = (
            (text_bot_right[0] - text_top_left[0]) // 2 - text_size[0][0] // 2,
            (text_bot_right[1] - text_top_left[1]) // 2 + text_size[0][1] // 2
        )

        text_position = (
            img[3] + text_offset[0],
            img[2] + text_offset[1]
        )

        cv2.rectangle(
            frame,
            text_top_left,
            text_bot_right,
            color,
            cv2.FILLED
        )

        cv2.putText(
            frame,
            name,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            .6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


class Ui_MainWindow(object):

    def __init__(self):

        super().__init__()
        self.previous_matches = []
        self.thread = VideoThread()
        self.thread.prepared_data.connect(self.update_state)
        self.thread.start()

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("Facial recognition")
        MainWindow.resize(1900, 720)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self._translate = QtCore.QCoreApplication.translate
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image_label = QtWidgets.QLabel(self.centralwidget)

        self.image_label.setGeometry(
            QtCore.QRect(0, 0, 1253, 720)
        )
        
        self.image_label.setObjectName("cam_section")
        self.header_title = QtWidgets.QLabel(self.centralwidget)

        self.header_title.setGeometry(
            QtCore.QRect(1257, 0, 521, 41)
        )

        self.header_title.setFont(font)
        self.header_title.setAlignment(QtCore.Qt.AlignCenter)
        self.header_title.setObjectName("header_title")
        self.header_2_title = QtWidgets.QLabel(self.centralwidget)

        self.header_2_title.setGeometry(
            QtCore.QRect(1786, 0, 141, 41)
        )

        self.header_2_title.setFont(font)
        self.header_2_title.setAlignment(QtCore.Qt.AlignCenter)
        self.header_2_title.setObjectName("header_title")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)

        self.tableWidget.setGeometry(
            QtCore.QRect(1257, 40, 521, 657)
        )

        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(self._translate("MainWindow", "Name"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(self._translate("MainWindow", "Date"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(self._translate("MainWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(self._translate("MainWindow", "Activity"))
        self.presence_table = QtWidgets.QTableWidget(self.centralwidget)

        self.presence_table.setGeometry(
            QtCore.QRect(1782, 40, 135, 657)
        )

        self.presence_table.setObjectName("presence_table")
        self.presence_table.setColumnCount(1)
        item = QtWidgets.QTableWidgetItem()
        self.presence_table.setHorizontalHeaderItem(0, item)
        item = self.presence_table.horizontalHeaderItem(0)
        item.setText(self._translate("MainWindow", "Name"))
        self.presence_table.itemClicked.connect(self.on_click)

        MainWindow.setCentralWidget(self.centralwidget)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")

        MainWindow.addToolBar(
            QtCore.Qt.TopToolBarArea,
            self.toolBar
        )

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.init_presence()

    def retranslateUi(self, MainWindow):

        MainWindow.setWindowTitle(
            self._translate("MainWindow", "MainWindow")
        
        )

        self.header_title.setText(
            self._translate("MainWindow", "Activity tracker:")
        )

        self.header_2_title.setText(
            self._translate("MainWindow", "Presence")
        )

    def log_person(self, name):

        db = TinyDB(os.path.join('./db.json'))
        person = Query()
        detection_record = db.search(person.name == name)
        isLeaving = len(detection_record) % 2 == 1
        current_date = datetime.today()

        person_log = {
            'name': name,
            'date': str(current_date.strftime('%d-%m-%Y')),
            'time': str(current_date.strftime('%H:%M:%S')),
            'activity': 'left room' if isLeaving else 'entered room'
        }

        self.add_entry(person_log)

        for row in range(self.presence_table.rowCount()):

            cell = self.presence_table.item(row, 0)

            if cell.text() == name:

                color = QtGui.QColor(0, 255, 0) \
                    if person_log['activity'] == 'entered room' \
                    else QtGui.QColor(255, 0, 0)

                cell.setBackground(color)

        db.insert(person_log)

    def add_entry(self, information):

        current_row = self.tableWidget.rowCount()
        data = list(information.values())
        self.tableWidget.insertRow(current_row)

        for col in range(4):
            cell = QtWidgets.QTableWidgetItem()
            cell.setText(data[col])
            cell.setFlags(QtCore.Qt.ItemIsEnabled)
            self.tableWidget.setItem(current_row, col, cell)


    def update_state(self, data):

        self.image_label.setPixmap(
            QPixmap.fromImage(data['frame'])
        )

        for name in data['matches']:
            self.log_person(name)


    def init_presence(self):
        global known_persons

        row = 0

        for name in known_persons:

            self.presence_table.insertRow(row)
            cell = QtWidgets.QTableWidgetItem()

            cell.setBackground(
                QtGui.QColor(255, 0, 0)
            )

            cell.setText(name)
            self.presence_table.setItem(row, 0, cell)
            row += 1

    def on_click(self, clicked_cell):

        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_SecondWindow(clicked_cell.text())
        self.ui.setupUi(self.window)
        self.window.show()


class Ui_SecondWindow(object):

    def __init__(self, name):
        self.name = name

    def setupUi(self, SecondWindow):

        SecondWindow.setObjectName("SecondWindow")
        SecondWindow.resize(790, 721)
        self.centralwidget = QtWidgets.QWidget(SecondWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 511, 721))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.name_select = QtWidgets.QComboBox(self.centralwidget)
        self.name_select.setGeometry(QtCore.QRect(630, 10, 151, 22))
        self.name_select.setObjectName("name_select")
        self.name_input_label = QtWidgets.QLabel(self.centralwidget)
        self.name_input_label.setGeometry(QtCore.QRect(520, 10, 51, 21))
        self.name_input_label.setObjectName("name_input_label")
        self.start_date_input_label = QtWidgets.QLabel(self.centralwidget)
        self.start_date_input_label.setGeometry(QtCore.QRect(520, 40, 71, 21))
        self.start_date_input_label.setObjectName("start_date_input_label")
        self.start_time_input_label = QtWidgets.QLabel(self.centralwidget)
        self.start_time_input_label.setGeometry(QtCore.QRect(520, 100, 61, 21))
        self.start_time_input_label.setObjectName("start_time_input_label")
        self.activity_input_label = QtWidgets.QLabel(self.centralwidget)
        self.activity_input_label.setGeometry(QtCore.QRect(520, 160, 51, 21))
        self.activity_input_label.setObjectName("activity_input_label")
        self.start_date_input = QtWidgets.QDateEdit(self.centralwidget)
        self.start_date_input.setGeometry(QtCore.QRect(630, 40, 151, 22))
        self.start_date_input.setObjectName("start_date_input")
        self.start_time_input = QtWidgets.QTimeEdit(self.centralwidget)
        self.start_time_input.setGeometry(QtCore.QRect(630, 100, 151, 22))
        self.start_time_input.setObjectName("start_time_input")
        self.activity_input = QtWidgets.QComboBox(self.centralwidget)
        self.activity_input.setGeometry(QtCore.QRect(630, 160, 151, 22))
        self.activity_input.setObjectName("activity_input")
        self.end_time_input = QtWidgets.QTimeEdit(self.centralwidget)
        self.end_time_input.setGeometry(QtCore.QRect(630, 130, 151, 22))
        self.end_time_input.setObjectName("end_time_input")
        self.end_time_input_label = QtWidgets.QLabel(self.centralwidget)
        self.end_time_input_label.setGeometry(QtCore.QRect(520, 130, 61, 21))
        self.end_time_input_label.setObjectName("end_time_input_label")
        self.end_date_input_label = QtWidgets.QLabel(self.centralwidget)
        self.end_date_input_label.setGeometry(QtCore.QRect(520, 70, 61, 21))
        self.end_date_input_label.setObjectName("end_date_input_label")
        self.end_date_input = QtWidgets.QDateEdit(self.centralwidget)
        self.end_date_input.setGeometry(QtCore.QRect(630, 70, 151, 22))
        self.end_date_input.setObjectName("end_date_input")
        self.filter_button = QtWidgets.QPushButton(self.centralwidget)
        self.filter_button.setGeometry(QtCore.QRect(690, 190, 93, 28))
        self.filter_button.setObjectName("filter_button")
        self.filter_button.clicked.connect(self.on_click)
        SecondWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(SecondWindow)
        QtCore.QMetaObject.connectSlotsByName(SecondWindow)
        self.show_activity()
        self.init_filters()

    def retranslateUi(self, SecondWindow):

        _translate = QtCore.QCoreApplication.translate
        SecondWindow.setWindowTitle(_translate("SecondWindow", "MainWindow"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("SecondWindow", "Name"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("SecondWindow", "Date"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("SecondWindow", "Time"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("SecondWindow", "Activity"))
        self.name_select.setItemText(0, _translate("SecondWindow", "name_select_item"))
        self.name_select.setItemText(1, _translate("SecondWindow", "name_select_item2"))
        self.name_input_label.setText(_translate("SecondWindow", "Name"))
        self.start_date_input_label.setText(_translate("SecondWindow", "Start date"))
        self.start_time_input_label.setText(_translate("SecondWindow", "Start time"))
        self.activity_input_label.setText(_translate("SecondWindow", "Activity"))
        self.end_time_input_label.setText(_translate("SecondWindow", "End time"))
        self.end_date_input_label.setText(_translate("SecondWindow", "End date"))
        self.filter_button.setText(_translate("SecondWindow", "Filter"))

    def init_filters(self):
        global names, activities

        for name in names:
            self.name_select.addItem(name)

        for activity in activities:
            self.activity_input.addItem(activity)

    def show_activity(self):

        db = TinyDB(os.path.join('./db.json'))
        person = Query()
        activity = db.search(person.name == self.name)

        for row in activity:
            self.add_row(row)

    def add_row(self, data):

        current_row = self.tableWidget.rowCount()
        data = list(data.values())
        self.tableWidget.insertRow(current_row)

        for col in range(4):
            cell = QtWidgets.QTableWidgetItem()
            cell.setText(data[col])
            cell.setFlags(QtCore.Qt.ItemIsEnabled)
            self.tableWidget.setItem(current_row, col, cell)

    def on_click(self):

        db = TinyDB(os.path.join('./db.json'))

        self.tableWidget.setRowCount(0)

        data = {
            'name': self.name_select.currentText(),
            'start_date': self.start_date_input.date().toPyDate().strftime('%d-%m-%Y'),
            'end_date': self.end_date_input.date().toPyDate().strftime('%d-%m-%Y'),
            'start_time': self.start_time_input.time().toPyTime(),
            'end_time': self.end_time_input.time().toPyTime(),
            'activity': self.activity_input.currentText()
        }
        
        entry = Query()

        filter_result = db.search(
            (entry.name == data['name'])
            &
            (data["start_date"] <= entry['date'] <= data["end_date"])
            &
            (str(data['start_time']) <= entry['time'] <= str(data["end_time"]))
            &
            (entry.activity == data['activity'])
        )

        for row in filter_result:
            self.add_row(row)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
