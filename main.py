import pygame as pg
import utils.utils as utils

class MediUI:
    """
    MediUI initializes a Pygame-based GUI for the AI Powered MediKit application.
    """

    def __init__(self):
        pg.init()
        self.display = pg.display
        self.width, self.height =  utils.get_screen_size()
        self.display.set_caption("AI Powered MediKit")
        self.size = (self.width, self.height)
        self.screen = self.display.set_mode(self.size, pg.SRCALPHA)  # Enable alpha for transparency
        
        self.bg_images, _ = utils.load_and_scale_images("./assets/bgframes", self.size)
        
        self.module_size = (int(self.width * 0.1), int(self.width * 0.1))
        self.module_images, self.module_img_paths = utils.load_and_scale_images("./assets/logos", self.module_size)
        self.module_img_pos = dict()
        
        self.bg_glass = pg.image.load("./assets/black-bg.png").convert_alpha()
        self.bg_glass = pg.transform.scale(self.bg_glass, (self.width//2 - (self.width//2)*0.26, self.height + self.height*0.4))
        self.bg_glass.set_alpha(200)  # Set transparency level

        self.select_img = pg.image.load("./assets/select.png").convert_alpha()
        self.select_img = pg.transform.scale(self.select_img, (self.module_size[0] + 40, self.module_size[1] + 40))

        self.font = pg.font.Font("./assets/quicksand.ttf", int(self.width*0.08))
        self.font_two = pg.font.Font("./assets/quicksand.ttf", 20)
        self.font_three = pg.font.Font("./assets/quicksand.ttf", 10)
        self.font_four = pg.font.Font("./assets/quicksand.ttf", 15)
        self.font_five = pg.font.Font("./assets/quicksand.ttf", 20)
        self.result_text = None
        self.features = utils.get_meta("features")
        self.site = "https://arihara-sudhan.github.io/AI-Powered-MediKit"
        self.classes = utils.get_meta("classes")
        self.should_open_site = True

    def blit_bg_image(self, img):
        self.screen.blit(img, (0, 0))

    def blit_module_images(self, images):
        x, y = 5, 5
        self.screen.blit(self.bg_glass, (-4, -90))  # Draw bg_glass with transparency
        for i, image in enumerate(images):
            self.screen.blit(image, (x, y))
            self.module_img_pos[self.module_img_paths[i]] = (x, y)
            x += int(self.width * 0.12)
            if (i + 1) % 3 == 0:
                y += int(self.width * 0.12)
                x = 5

    def blit_select_img(self, x, y):
        self.screen.blit(self.select_img, (x - 20, y - 20))

    def render_text(self, text, position):
        """
        Render text on the screen.
        """
        text_surface = self.font.render(text, True, (50, 106, 174))
        self.screen.blit(text_surface, position)
        text_surface2 = self.font_two.render("AI POWERED", True, (50, 106, 174))
        self.screen.blit(text_surface2, (position[0] + 100, position[1] - 6))

    def render_description(self, key, text, position):
        """
        Render description of a module
        """
        key_surface = self.font_four.render(key.upper(), True, (255, 255, 255))
        key_rect = key_surface.get_rect(topleft=position)
        background_rect = pg.Surface((self.module_size[0], self.module_size[1] - 90))
        background_rect.fill((0, 0, 0))

        self.screen.blit(background_rect, key_rect.topleft)
        self.screen.blit(key_surface, position)

        text_surface = self.font_three.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(topleft=(position[0], position[1] + 20))

        self.screen.blit(background_rect, text_rect.topleft)
        self.screen.blit(text_surface, (position[0], position[1] + 20))

    def mousepos_process(self, mx, my):
        for key, pos in self.module_img_pos.items():
            key_name = key.split("/")[-1].split(".")[0]  # sanitize
            if pos[0] - 20 <= mx <= pos[0] + 80 and pos[1] - 20 <= my <= pos[1] + 80:
                self.blit_select_img(pos[0], pos[1])
                if key_name in self.features:
                    self.render_description(key_name, self.features[key_name], pos)
                return key_name
        return None

def main():
    med_ui = MediUI()
    restext = 0
    res_text_rect = None

    APP_STATE = True
    clock = pg.time.Clock()

    bg_frame = 0
    total_frames = len(med_ui.bg_images)
    frame_rate = 30
    bg_frame_direction = 1
    clicked_module = None

    while APP_STATE:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                APP_STATE = False
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    utils.pause_audio()
            elif event.type == pg.MOUSEBUTTONDOWN:
                if res_text_rect and res_text_rect.collidepoint(event.pos):
                    if clicked_module and med_ui.should_open_site:
                        utils.open_site(med_ui.site +"/#"+clicked_module)

                mx, my = pg.mouse.get_pos()
                clicked_module = med_ui.mousepos_process(mx, my)
                if clicked_module:
                    print(clicked_module)
                    if clicked_module == "herb":
                        user_prompt = utils.get_text_from_user()
                        if user_prompt:
                            metadata = utils.load_json_data("meta/herbs.json")
                            closest = utils.find_closest_record(user_prompt, metadata)
                            med_ui.result_text = f"{closest['name']} can be used to cure this condition!"
                            restext = 1
                            med_ui.should_open_site = False
                            utils.show_image(closest["img_path"], "Herbal Solution", closest["name"])

                    elif clicked_module == "heartbeat":
                        audio_file_path = utils.browse_file("Audio Files (*.mp3 *.wav)")
                        if audio_file_path:
                            label = utils.classify_audio(audio_file_path)
                            if label:
                                med_ui.result_text = label
                                restext = 1
                                med_ui.should_open_site = True
                                utils.play_audio(audio_file_path)
                            else:
                                print("SOMETHING WENT WRONG")
                    elif clicked_module == "tablet":
                        utils.fewshot()

                    elif clicked_module == "chatbot":
                        utils.init_chatbot()

                    else:
                        image_file_path = utils.browse_file()
                        if image_file_path:
                            result = utils.classify_img(clicked_module, image_file_path, med_ui.classes[clicked_module]["num_classes"])
                            if type(result)==int:
                                med_ui.result_text = med_ui.classes[clicked_module]["labels"][str(result)]
                                med_ui.should_open_site = True
                                utils.show_image(image_file_path, clicked_module, med_ui.result_text)
                            else:
                                med_ui.result_text = "Model can't classify this!"
                        restext = 1
    
        if med_ui.bg_images:
            med_ui.blit_bg_image(med_ui.bg_images[int(bg_frame)])
            bg_frame += bg_frame_direction
            if bg_frame >= total_frames - 1 or bg_frame <= 0:
                bg_frame_direction *= -1

        if med_ui.module_images:
            med_ui.blit_module_images(med_ui.module_images)

        med_ui.render_text("MediKit", (med_ui.width // 2 + 70, 20))

        mx, my = pg.mouse.get_pos()
        med_ui.mousepos_process(mx, my)

        if restext:
            res_text_surface = med_ui.font_five.render(med_ui.result_text, True, (255, 255, 255))
            res_text_rect = res_text_surface.get_rect(center=(med_ui.width//2+230, med_ui.height-80))
            med_ui.screen.blit(res_text_surface, res_text_rect)
            restext += 1
            if restext == 250:
                restext = False
                res_text_rect = None
        
        pg.display.flip()
        clock.tick(frame_rate)

    pg.quit()

main()