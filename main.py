import cv2 
from Architecture import ModelRunners
from PyQt5.QtWidgets import (QApplication,QWidget,QVBoxLayout,QLabel,
                            QPushButton,QFileDialog,QErrorMessage,QCheckBox,QLineEdit)
import sys 
modelrunners = ModelRunners('Model/NDM.pth')

class RealTimeCamera(QWidget) :
    def __init__(self) :
        super().__init__()
        self.setWindowTitle("RealtimeCamera")
        layout = QVBoxLayout()
        self.hight_line_edit = QLineEdit()
        self.hight_line_edit.setPlaceholderText("Type hight as integer : 0")
        self.weidth_line_edit = QLineEdit()
        self.weidth_line_edit.setPlaceholderText("Type weidth as integere : 0")
        self.button_confir_size_edited = QPushButton("submit size")
        self.button_play = QPushButton("Run Camera")
        self.web_cam_mode = 0 
        self.hight_error_editting = QErrorMessage(self)
        self.weidth_error_editting = QErrorMessage(self)
        self.__web_cam_size = (720,480)
        self.button_confir_size_edited.clicked.connect(self.confirmed_editting_size)
        self.button_play.clicked.connect(self.run_cammera)
        self.checkbox = QCheckBox("use External Camera ")
        self.__web_cam = 0 
        self.error_camera = QErrorMessage(self)
        self.checkbox.stateChanged.connect(self.change_to_extern)
        layout.addWidget(self.button_play)
        layout.addWidget(self.hight_line_edit)
        layout.addWidget(self.weidth_line_edit)
        layout.addWidget(self.button_confir_size_edited)
        layout.addWidget(self.checkbox)
        self.setLayout(layout)
    
    def confirmed_editting_size (self) :
        if int(self.hight_line_edit.text()) :
            hight = int(self.hight_line_edit.text())
        else :
            self.hight_error_editting.showMessage("Error hight must be number as integer like (0/9)")
        
        if int(self.weidth_line_edit.text()) :
            weidth = int(self.weidth_line_edit.text())
        else :
            self.weidth_error_editting.showMessage("Error weidth must be number as integer like (0/9)")

        self.__web_cam_size= (weidth,hight)

    
    def change_to_extern (self) :
        self.__web_cam = 1 
        if self.checkbox.isChecked() is False  :
            self.__web_cam = 0 
        
    
    def run_cammera(self):
        capture = cv2.VideoCapture(self.__web_cam)
        if not capture.isOpened():
            self.error_camera.showMessage("Error External Camera not available \n set to default camera")
            capture = cv2.VideoCapture(0)

        while True:
            ret, frame = capture.read()
            if not ret:
                break 

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img_rgb, (384,384))
            predicted = modelrunners.modelrun(img_resize)

            cv2.putText(frame, f"not Noice range : {predicted[0][0] * 100}%", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Noise low range : {predicted[0][1] * 100} %", (10,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (13,255,232), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Noise hight range : {predicted[0][2] * 100}%", (10,70),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,244), 2, cv2.LINE_AA)
            frame = cv2.resize(frame, self.__web_cam_size)
            cv2.imshow("RealTime Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()
    
    

class RecordedVideosRunner(QWidget) :
    def __init__ (self):
        super().__init__()
        self.setWindowTitle("RecordedVideosRunners")
        self.scan_Noice = QPushButton("Scan Noice")
        self.submit_videos = QPushButton("Submit Video")
        self.Error_File = QErrorMessage(self)
        self.Error_video_runner = QErrorMessage(self)
        self.hight_editor = QLineEdit()
        self.hight_editor.setPlaceholderText("input hight as integer (0/9)")
        self.width_editor = QLineEdit()
        self.width_editor.setPlaceholderText("input width as integer (0/9)")
        self.submit_video_size = QPushButton("submit size")
        self.submit_video_size.clicked.connect(self.submit_size)
        self.__web_cam_size = (720,480)
        self.__videos_target = None 
        self.submit_videos.clicked.connect(self.get_file_addreas)
        self.scan_Noice.clicked.connect(self.scan_videos)
        layout = QVBoxLayout()
        layout.addWidget(self.submit_videos)
        layout.addWidget(self.hight_editor)
        layout.addWidget(self.width_editor)
        layout.addWidget(self.submit_video_size)
        layout.addWidget(self.scan_Noice)
        self.setLayout(layout)
    
    def get_file_addreas (self) :
        path = QFileDialog.getOpenFileName(self,"Choice Videos File")
        if path :
            self.__videos_target = path[0]
        else :
            self.Error_File.showMessage("Error File is not found")
        
    def submit_size (self) :
        if int(self.hight_editor.text()) :
            hight = int(self.hight_editor.text())
        else : 
            QErrorMessage(self).showMessage("Error the hight must be integer number (0/9)")
        
        if int(self.width_editor.text()) :
            width = int(self.hight_editor.text())
        else :
            QErrorMessage(self).showMessage("Error the width must be integer number (0/9)")
        
        self.__web_cam_size = (hight,width)

    
    def scan_videos (self) :
        if self.__videos_target is None :
            self.Error_video_runner.showMessage("Error file is not found")
        capture = cv2.VideoCapture(self.__videos_target)
        
        while (True) :
            ret,frame = capture.read()
            if not ret :
                break 

            rgb_input = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            tensor_input = cv2.resize(rgb_input,(384,384))
            predicted = modelrunners.modelrun(tensor_input)
            cv2.putText(frame, f"not Noice range : {predicted[0][0] * 100}%", (10,25),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Noise low range : {predicted[0][1] * 100} %", (10,50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (13,255,232), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Noise hight range : {predicted[0][2] * 100}%", (10,75),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,244), 2, cv2.LINE_AA)
            
            frame = cv2.resize(frame,self.__web_cam_size)
            cv2.imshow("Recorded Videos",frame)

            if cv2.waitKey(1) & 0xFF == ord('q') :
                break 
        
        capture.release()
        cv2.destroyAllWindows()

class Noice_Detector (QWidget) :
    def __init__ (self) :
        super().__init__()
        self.realtime_camera_tools = RealTimeCamera()
        self.recorder_tools = RecordedVideosRunner()
        self.realtime_button = QPushButton("Scan by RealTime Camera")
        self.recorded_videos = QPushButton("Scan by recorded Videos")
        self.realtime_button.clicked.connect(self.real_time_runners)
        self.recorded_videos.clicked.connect(self.recorded_runners)
        self.label = QLabel("Model detail : \n Name : NDM (Noice Detector Model) \n size : 6.5 mb \n type : Vision Transformers")
        layout = QVBoxLayout()
        layout.addWidget(self.realtime_button)
        layout.addWidget(self.recorded_videos)
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def real_time_runners (self) :
        self.realtime_camera_tools.show()
    
    def recorded_runners (self) :
        self.recorder_tools.show()




if __name__ == "__main__" :

    main_app = QApplication(sys.argv)
    main_gui = Noice_Detector()
    main_gui.show()
    sys.exit(main_app.exec_())