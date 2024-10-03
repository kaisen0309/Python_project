# -*- coding: utf-8 -*-
import dlib
import cv2
import pygame
import sys
import threading
import time
import random
import speech_recognition as sr
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        # 基本設定
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # self.cap = cv2.VideoCapture(0)

        # 存上一貞位置(list)
        self.previous_landmarks = None
        # 抬多少單位算抬頭
        self.head_constant = 50
        #維持後不動的誤差範圍的單位
        self.interval = 25
        # 抬頭狀態
        self.head_state = "none"
        # 設起始位置
        self.x_initial = None
        self.y_initial = None
        #計算偵測的次數
        self.count_landmark = 0
        #幾次count_landmark 後check一次
        self.count_landmark_constant = 45

        # self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        # 跳躍訊號
        self.dog_jump_signal = False
        self.dog_down_signal = False
        self.dog_right_signal = False
        self.dog_left_signal = False

    def detect_face(self):
        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,  # 新版
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)
                    
                    # 偵測現在的位置跟上一貞的差別
                    #(目前previous只剩畫箭頭用)
                    if self.previous_landmarks:
                        for index, landmark in enumerate(face_landmarks.landmark):  # index是索引值(即0~467 468個特貞點) #landmark才是座標(歸一化)
                            x_previous, y_previous = self.previous_landmarks[index]  # 去previous_landmarks抓相同index來做比較

                            # landmark.x y 是相對於圖形的長寬的歸一化座標 所以*image的長([0])寬([1]) 才是實際上正確的座標
                            x_current = int(landmark.x * image.shape[1])
                            y_current = int(landmark.y * image.shape[0])

                            # draw arrow
                            cv2.arrowedLine(image, (x_previous, y_previous), (x_current, y_current), (0, 255, 0), 2, tipLength=0.5)

                        # 先抓1號特偵點測試(鼻子)
                        face_index = 1
                        # x_previous, y_previous = previous_landmarks[face_index]
                        x_current = int(face_landmarks.landmark[face_index].x * image.shape[1])
                        y_current = int(face_landmarks.landmark[face_index].y * image.shape[0])

                        #固定init 位置，之後的位移已都以這格座標來調整
                        if self.x_initial is None:
                            self.x_initial = x_current
                        if self.y_initial is None:
                            self.y_initial = y_current
                        #給使用者中心點位置
                        cv2.putText(image, "*", (self.x_initial, self.y_initial), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                        # 更新head_state
                        if (self.y_initial - y_current) >= self.head_constant:
                            self.head_state = "Up"
                            self.dog_jump_signal = True
                        elif (y_current - self.y_initial) >= self.head_constant:
                            self.head_state = "Down"
                            self.dog_down_signal = True
                        elif (x_current - self.x_initial) >= self.head_constant:
                            self.head_state = "Right"
                            self.dog_right_signal = True
                        elif (self.x_initial - x_current) >= self.head_constant:
                            self.head_state = "Left"
                            self.dog_left_signal = True
                        #偵測次數計算
                        self.count_landmark += 1
                        #每偵測constant次時check他離中心點的位置，若<interval 則算none(沒上下左右的意思)
                        if self.count_landmark > self.count_landmark_constant and abs(self.x_initial - x_current)<self.interval and abs(self.y_initial - y_current)<self.interval:
                            self.head_state = "none"
                            #如果none就reset count_landmark，怕到時候值太大
                            self.count_landmark = 0
                        cv2.putText(image, self.head_state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                    # 更新上一貞位置(即目前此刻的位置 下一刻就跟這個比較)(只剩畫箭頭用)
                    self.previous_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in face_landmarks.landmark]

            cv2.imshow('MediaPipe FaceMesh', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        self.cap.release()
        cv2.destroyAllWindows()    

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

class SpeakListener:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.keywords = ["借過", "阿", "走開","嗨","你好"]
        # 開啟狀態
        self.load = True
        # 聲音訊號
        self.dog_yell_signal = False
    
    def detect_speak(self):
        while self.load:
            try:
                with self.microphone as source:
                    # 為了調整麥克風環境噪音的音量
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    print("正在聽...")
                    # 縮短錄音時間
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # 使用 Google 語音識別
                result = self.recognizer.recognize_google(audio, language="zh-TW")
                print(f"你說了: {result}")
                for keyword in self.keywords:
                    if keyword in result:
                    # 偵測到關鍵字，開啟訊號
                        self.dog_yell_signal = True
                        print("signal true")
            except sr.UnknownValueError:
                print("無法識別的語音")
            except sr.RequestError as e:
                print(f"語音識別服務出現錯誤: {e}")
            except sr.WaitTimeoutError:
                print("錄音超時，未檢測到語音")
    def close(self):
        self.load = False
        
class Truck:
    def __init__(self,h,w):
        self.truck_image = None
        self.level = 0
        self.truck_image = pygame.image.load('truck.png')
        self.truck_image = pygame.transform.scale(self.truck_image, (100, 70))
        self.truck_y = random.randint(0, h - self.truck_image.get_height())
        self.position = pygame.Rect(w, self.truck_y, self.truck_image.get_width(), self.truck_image.get_height())
    def truck2(self):
        truckbroke_image = pygame.image.load('truckbroke.png')
        self.truck_image = pygame.transform.scale(truckbroke_image, (100, 70))
        self.level = 1
        
class DogGame:
    def __init__(self):
        pygame.init()
        self.screen_width, self.screen_height = 1024, 568  # 更大的屏幕尺寸
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flying Dog Game")
        
        # 圖片
        self.background = pygame.image.load('road.png')
        self.background = pygame.transform.scale(self.background, (self.screen_width, self.screen_height))
        
        dog_image_1 = pygame.image.load('dog.png')
        dog_image_2 = pygame.image.load('dogrun.png')
        self.dog_images = [
            pygame.transform.scale(dog_image_1, (60, 50)),
            pygame.transform.scale(dog_image_2, (60, 50))
        ]
        
        car_image_blue = pygame.image.load('car.png')
        car_image_red = pygame.image.load('redcar.png')
        self.car_image_blue = pygame.transform.scale(car_image_blue, (80, 50))
        self.car_image_red = pygame.transform.scale(car_image_red,(80,50))

        # 小狗大小
        self.dog_width, self.dog_height = 60, 50
        self.dog_x = 100
        self.dog_y = self.screen_height // 2
        self.dog_y_velocity = 2
        self.flap_power = -10
        self.dog_frame = 0
        self.dog_speed=10
       
        self.font = pygame.font.SysFont(None, 55)
        self.clock = pygame.time.Clock()
        self.running = True
        
        # 障礙物
        self.obstacles = []
        self.obstacle_speed = 3

        # 障礙物卡車
        self.trucks = []
        self.truck_speed = 1

        
        # 骨頭
        self.bone_image = pygame.image.load('bone.png')
        self.bone_image = pygame.transform.scale(self.bone_image, (40, 25))
        self.bones = []

         # 黃金骨頭
        self.goldbone_image = pygame.image.load('goldbone.png')
        self.goldbone_image = pygame.transform.scale(self.goldbone_image, (40, 25))
        self.goldbones = []

        # 分數
        self.score = 0
        self.lastTicks = pygame.time.get_ticks()
        self.bone_score = 0

        # 無敵
        self.invincible = False
        self.invincible_start_time = None
        self.remaining_time = -1

        # 計時器
        self.timer_active = False
        self.timer_start_time = None
        self.invincible_timer = None

        # 臉部偵測
        self.face_detector = FaceDetector()
        # 聲音偵測
        self.speak_detector = SpeakListener()
        
    def create_obstacle(self):
        obstacle_y = random.randint(0, self.screen_height - self.car_image_blue.get_height())
        obstacle = pygame.Rect(self.screen_width, obstacle_y, self.car_image_blue.get_width(), self.car_image_blue.get_height())
        self.obstacles.append(obstacle)

    def create_bone(self):
            bone_y = random.randint(0, self.screen_height - self.bone_image.get_height())
            bone = pygame.Rect(self.screen_width, bone_y, self.bone_image.get_width(), self.bone_image.get_height())
            self.bones.append(bone)

    def create_goldbone(self):
        goldbone_y = random.randint(0, self.screen_height - self.goldbone_image.get_height())
        goldbone = pygame.Rect(self.screen_width, goldbone_y, self.goldbone_image.get_width(), self.goldbone_image.get_height())
        self.goldbones.append(goldbone)

    def create_truck(self):
        truck = Truck(self.screen_height, self.screen_width)
        self.trucks.append(truck)
        
    def display_timer(self, remaining_time):
        timer_text = self.font.render(f"{int(remaining_time)}", True, (0, 0, 0))  # 黑色字體
        timer_text_rect = timer_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))  # 放置在屏幕中央
        self.screen.blit(timer_text, timer_text_rect)

    def show_game_over_screen(self):
        self.screen.fill((255, 255, 255))
        game_over_text = self.font.render("Game Over", True, (255, 0, 0))
        self.screen.blit(game_over_text, (self.screen_width // 2 - game_over_text.get_width() // 2,
                                          self.screen_height // 2 - game_over_text.get_height() // 2))
        pygame.display.flip()
        # waiting_for_input = True
        time.sleep(1)

    # 遊戲主程式運行區
    def run(self):
        # 多執行序同時執行臉部偵測class中的detect_face()，開始偵測臉部
        threading.Thread(target=self.face_detector.detect_face).start()
        # 多執行序同時執行聲音偵測class中的detect_speak()，開始偵測聲音
        threading.Thread(target=self.speak_detector.detect_speak).start()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                # 鍵盤操作
                if event.type == pygame.KEYDOWN:
                    self.dog_y += self.flap_power
            # 若跳躍信號為開啟，代表嘴巴正在張開
            if self.face_detector.dog_jump_signal:
                # 往上跑並關閉訊號
                self.dog_y += self.flap_power
                self.face_detector.dog_jump_signal = False
            elif self.face_detector.dog_down_signal:
                self.dog_y -= self.flap_power
                self.face_detector.dog_down_signal = False
            elif self.face_detector.dog_right_signal:
                self.dog_x -= self.flap_power
                self.face_detector.dog_right_signal = False
            elif self.face_detector.dog_left_signal:
                self.dog_x += self.flap_power
                self.face_detector.dog_left_signal = False

            # else:
            #     # 往下掉
            #     self.dog_y += self.dog_y_velocity
            

            # 碰邊界不結束但不能繼續走
            if self.dog_y < 0:
                self.dog_y = 0
            elif self.dog_y + self.dog_height > self.screen_height:
                self.dog_y = self.screen_height - self.dog_height
            
            if self.dog_x < 0:
                self.dog_x = 0
            elif self.dog_x + self.dog_width > self.screen_width:
                self.dog_x = self.screen_width - self.dog_width

            self.screen.blit(self.background, (0, 0))

            self.dog_frame = (self.dog_frame + 1) % 2
            self.screen.blit(self.dog_images[self.dog_frame], (self.dog_x, self.dog_y))
            
            # 生成的機率
            if random.randint(1, 100) < 3:
                self.create_obstacle()
            if random.randint(1, 100) < 2:
                self.create_bone()
            if random.randint(1, 500) < 6:
                self.create_goldbone()
            if random.randint(1, 500) < 2:
                self.create_truck()

            for ob in self.obstacles:
                ob.x -= self.obstacle_speed
                if ob.x < 0:
                    self.obstacles.remove(ob)
                if not self.invincible and ob.colliderect(pygame.Rect(self.dog_x, self.dog_y, self.dog_width, self.dog_height)):
                    # 結束畫面
                    self.show_game_over_screen()
                    self.running = False

            for bone in self.bones:
                bone.x -= self.obstacle_speed
                if bone.x < 0:
                    self.bones.remove(bone)

                if bone.colliderect(pygame.Rect(self.dog_x, self.dog_y, self.dog_width, self.dog_height)):
                    self.bones.remove(bone)
                    self.bone_score += 5

            for goldbone in self.goldbones:
                goldbone.x -= self.obstacle_speed
                if goldbone.x < 0:
                    self.goldbones.remove(goldbone)

                if goldbone.colliderect(pygame.Rect(self.dog_x, self.dog_y, self.dog_width, self.dog_height)):
                    self.goldbones.remove(goldbone)

                    # 開始計時
                    self.invincible = True
                    self.invincible_start_time = pygame.time.get_ticks()

            for truck in self.trucks:
                truck.position.x -= self.truck_speed
                if truck.position.x < 0:
                    self.trucks.remove(truck)

                if truck.position.colliderect(pygame.Rect(self.dog_x, self.dog_y, self.dog_width, self.dog_height)):
                    # 結束畫面
                    self.show_game_over_screen()
                    self.running = False
            # 聲音檢查
            if self.speak_detector.dog_yell_signal:
                for truck in self.trucks:
                    if truck.level == 0:
                        truck.truck2()
                    else:
                        self.trucks.remove(truck)
                        print("remove truck")
                self.speak_detector.dog_yell_signal = False

            # 無敵時間檢查
            if self.invincible:
                elapsed_time = (pygame.time.get_ticks() - self.invincible_start_time) / 1000
                self.remaining_time = 5 - elapsed_time
            
            # 重新繪製畫面
            self.screen.blit(self.background, (0, 0))
            if self.remaining_time <= 0:
                self.invincible = False
            else:
                invincible_timer_text = self.font.render(f"Invincible: {self.remaining_time:.1f}s", True, (255, 0, 0))
                self.screen.blit(invincible_timer_text, (self.screen_width - invincible_timer_text.get_width() - 20, 20))
            
            for ob in self.obstacles:
                if self.invincible: 
                    self.screen.blit(self.car_image_red, (ob.x, ob.y))
                else: 
                    self.screen.blit(self.car_image_blue, (ob.x, ob.y)) 

            for bone in self.bones:
                self.screen.blit(self.bone_image, (bone.x, bone.y))

            for goldbone in self.goldbones:
                self.screen.blit(self.goldbone_image, (goldbone.x, goldbone.y))
            
            for truck in self.trucks:
                self.screen.blit(truck.truck_image, (truck.position.x, truck.position.y))
            
                # if self.speak_detector.dog_yell_signal:
                #     for truck in self.trucks:
                #         self.screen.blit(self.truckbroke_image, (truck.x, truck.y))
                #     self.speak_detector.dog_yell_signal = False

            self.screen.blit(self.dog_images[self.dog_frame], (self.dog_x, self.dog_y))

            # 分數計算
            temp=pygame.time.get_ticks()
            self.score+=temp-self.lastTicks
            self.lastTicks = temp

            scoreObject = self.font.render("score: %s"%(int(self.score/1000+ self.bone_score)), True, (255, 0, 0))
            self.screen.blit(scoreObject,(0,0))
            
            pygame.display.flip()
            self.clock.tick(30)
        
        pygame.quit()
        # 關閉臉部辨識
        threading.Thread(target=self.face_detector.close).start()
        # 關閉聲音辨識
        threading.Thread(target=self.speak_detector.close).start()
        
        sys.exit()

if __name__ == "__main__":
    DogGame().run()
