"""A collection of all commands that Jett can use to interact with the game."""

from src.common import config, settings, utils
import time
import math
from src.routine.components import Command
from src.common.vkeys import press, key_down, key_up


# List of key mappings
class Key:
    # Movement
    JUMP = 'space'
    
    # Skills
    TWIN_STAR_ATTACK = 'ctrl'

    # Buffs
    CHARGE = 'w'

#########################
#       Commands        #
#########################
def step(direction, target):
    """
    Performs one movement step in the given DIRECTION towards TARGET.
    Should not press any arrow keys, as those are handled by Auto Maple.
    """

    num_presses = 2
    if direction == 'up' or direction == 'down':
        num_presses = 1
    if config.stage_fright and direction != 'up' and utils.bernoulli(0.75):
        time.sleep(utils.rand_float(0.1, 0.3))
    d_y = target[1] - config.player_pos[1]
    if abs(d_y) > settings.move_tolerance * 1.5:
        if direction == 'down':
            press(Key.JUMP, 3)
        elif direction == 'up':
            press(Key.JUMP, 1)
    # press(Key.FLASH_JUMP, num_presses)
import cv2
class MoveAndAttack(Command):
    """移動到指定位置，並在移動途中偵測怪物並攻擊。"""
    def __init__(self, x, y, max_steps=15, forward_detection_width_multiplier=1.0, backward_detection_width_multiplier=0.5):
        super().__init__(locals())
        self.target = (float(x), float(y))
        self.max_steps = settings.validate_nonnegative_int(max_steps)
        self.forward_detection_width_multiplier = float(forward_detection_width_multiplier)
        self.backward_detection_width_multiplier = float(backward_detection_width_multiplier)
        self.prev_direction = ''
        print(f"前方: {self.forward_detection_width_multiplier}\n")
        print(f"後方: {self.backward_detection_width_multiplier}\n")

    def _new_direction(self, new):
        key_down(new)
        print(f"new_direction，按下: {new}\n")
        if self.prev_direction and self.prev_direction != new:
            key_up(self.prev_direction)
            print(f"new_direction，取消按: {self.prev_direction}\n")
        self.prev_direction = new

    def main(self):
        print("[DEBUG] MoveAndAttack: main() started")
        counter = self.max_steps
        path = config.layout.shortest_path(config.player_pos, self.target)
        print(f"[DEBUG] MoveAndAttack: path = {path}")
        for i, point in enumerate(path):
            toggle = True
            self.prev_direction = ''
            local_error = utils.distance(config.player_pos, point)
            global_error = utils.distance(config.player_pos, self.target)
            while config.enabled and counter > 0 and \
                    local_error > settings.move_tolerance and \
                    global_error > settings.move_tolerance:
                if toggle:
                    d_x = point[0] - config.player_pos[0]
                    if abs(d_x) > settings.move_tolerance / math.sqrt(2):
                        if d_x < 0:
                            key = 'left'
                        else:
                            key = 'right'
                        self._new_direction(key)
                        step(key, point)
                        # 偵測怪物
                        if config.mob_pos and config.player_pos_in_attack_region:
                            while True and config.enabled:
                                print(f"有怪在螢幕範圍\n")
                                # 定義偵測範圍
                                player_width = config.player_pos_in_attack_region[0][1][0] - config.player_pos_in_attack_region[0][0][0]
                                player_x = config.player_pos_in_attack_region[0][0][0] + player_width / 2
                                if key == 'right':
                                    detection_start = player_x - player_width * self.backward_detection_width_multiplier
                                    detection_end = player_x + player_width * self.forward_detection_width_multiplier
                                elif key == 'left':
                                    detection_start = player_x - player_width * self.forward_detection_width_multiplier
                                    detection_end = player_x + player_width * self.backward_detection_width_multiplier

                                # 篩選在偵測範圍內的怪物，且y軸有重疊
                                valid_mobs = []
                                player_y0 = config.player_pos_in_attack_region[0][0][1]
                                player_y1 = config.player_pos_in_attack_region[0][1][1]
                                for mob in config.mob_pos:
                                    mob_x0 = mob[0][0]
                                    mob_x1 = mob[1][0]
                                    mob_y0 = mob[0][1]
                                    mob_y1 = mob[1][1]
                                    if detection_start <= mob_x0 <= detection_end or detection_start <= mob_x1 <= detection_end:
                                        if max(player_y0, mob_y0) < min(player_y1, mob_y1): # 檢查y軸是否重疊
                                            valid_mobs.append(mob)
                                
                                print(f"在設定的偵測範圍內怪: {len(valid_mobs)}\n")
                                # 找到最近的怪物
                                nearest_mob = None
                                min_distance = float('inf')

                                for mob in valid_mobs:
                                    distance = abs(player_x - (mob[0][0] + mob[1][0]) / 2) # 計算怪物中心點的距離
                                    if distance < min_distance:
                                        min_distance = distance
                                        nearest_mob = mob
                                
                                if not valid_mobs:
                                    break

                                if nearest_mob:
                                    key_up('left')
                                    key_up('right')
                                    # 面向怪物
                                    if nearest_mob[0][0] > config.player_pos_in_attack_region[0][0][0]: # 判斷怪物是否在玩家右邊
                                        key_down('right')
                                        time.sleep(0.05)
                                        key_up('right')
                                    elif nearest_mob[0][0] < config.player_pos_in_attack_region[0][0][0]: # 判斷怪物是否在玩家左邊
                                        key_down('left')
                                        time.sleep(0.05)
                                        key_up('left')
                                    # 攻擊最近的怪物
                                    TwinStarAttack().main()
                                    time.sleep(0.1)
                                
                            self._new_direction(key)
                        if settings.record_layout:
                            config.layout.add(*config.player_pos)
                        counter -= 1
                        if i < len(path) - 1:
                            time.sleep(0.15)
                else:
                    d_y = point[1] - config.player_pos[1]
                    if abs(d_y) > settings.move_tolerance / math.sqrt(2):
                        if d_y < 0:
                            key = 'up'
                        else:
                            key = 'down'
                        self._new_direction(key)
                        step(key, point)
                        
                        if settings.record_layout:
                            config.layout.add(*config.player_pos)
                        counter -= 1
                        if i < len(path) - 1:
                            time.sleep(0.05)
                local_error = utils.distance(config.player_pos, point)
                global_error = utils.distance(config.player_pos, self.target)
                toggle = not toggle
            if self.prev_direction:
                key_up(self.prev_direction)


class Adjust(Command):
    """Fine-tunes player position using small movements."""

    def __init__(self, x, y, max_steps=5):
        super().__init__(locals())
        self.target = (float(x), float(y))
        self.max_steps = settings.validate_nonnegative_int(max_steps)

    def main(self):
        counter = self.max_steps
        toggle = True
        error = utils.distance(config.player_pos, self.target)
        while config.enabled and counter > 0 and error > settings.adjust_tolerance:
            if toggle:
                d_x = self.target[0] - config.player_pos[0]
                threshold = settings.adjust_tolerance / math.sqrt(2)
                if abs(d_x) > threshold:
                    walk_counter = 0
                    if d_x < 0:
                        key_down('left')
                        while config.enabled and d_x < -1 * threshold and walk_counter < 60:
                            time.sleep(0.05)
                            walk_counter += 1
                            d_x = self.target[0] - config.player_pos[0]
                        key_up('left')
                    else:
                        key_down('right')
                        while config.enabled and d_x > threshold and walk_counter < 60:
                            time.sleep(0.05)
                            walk_counter += 1
                            d_x = self.target[0] - config.player_pos[0]
                        key_up('right')
                    counter -= 1
            else:
                d_y = self.target[1] - config.player_pos[1]
                if abs(d_y) > settings.adjust_tolerance / math.sqrt(2):
                    if d_y < 0:
                        key_down('up')
                        press(Key.JUMP, 2)
                        time.sleep(0.05)
                        press(Key.JUMP, 3, down_time=0.1)
                        key_up('up')
                        time.sleep(0.05)
                    else:
                        key_down('down')
                        time.sleep(0.05)
                        press(Key.JUMP, 3, down_time=0.1)
                        key_up('down')
                        time.sleep(0.05)
                    counter -= 1
            error = utils.distance(config.player_pos, self.target)
            toggle = not toggle

class Buff(Command):
    """Uses each of Jett's buffs once."""

    def __init__(self):
        super().__init__(locals())
        self.charge_buff_time = 0
        self.charge_cd = 60

    def main(self):
        now = time.time()
        if self.charge_buff_time == 0 or now - self.charge_buff_time > self.charge_cd:
            press(Key.CHARGE, 3, up_time=0.3)
            self.charge_buff_time = now

class ClimbRope(Command):
    """ClimbRope."""

    def __init__(self, start_x, start_y, end_x, end_y, jump_rope_tolerance = 0.01):
        super().__init__(locals())
        self.rope_start = (float(start_x), float(start_y))
        self.rope_end = (float(end_x), float(end_y))
        self.jump_rope_tolerance = float(jump_rope_tolerance)

    def main(self):
        print(f"[DEBUG] ClimbRope: rope_start = {self.rope_start}, rope_end = {self.rope_end}, jump_rope_tolerance = {self.jump_rope_tolerance}")
        key_down('up')
        print("[DEBUG] ClimbRope: Key down 'up'")
        
        #是否已爬超過繩子一半(至少上繩子或是已爬完)
        while config.enabled and (config.player_pos[1] > ((self.rope_start[1]- self.rope_end[1])/2 +  self.rope_end[1])) :
            mid_rope = (self.rope_start[1] - self.rope_end[1]) / 2 + self.rope_end[1]
            condition = config.player_pos[1] > mid_rope

            print(f"player_pos[1]: {config.player_pos[1]}")
            print(f"rope_start[1]: {self.rope_start[1]}, rope_end[1]: {self.rope_end[1]}")
            print(f"mid_rope: {mid_rope}")
            print(f"Condition result: {condition}")
            d_x = self.rope_start[0] - config.player_pos[0]

            # 走到繩子起點
            while True:
                print(f"[DEBUG] ClimbRope: d_x = {d_x}")
                
                if d_x < 0:
                    key_down('left')
                    print("[DEBUG] ClimbRope: Key down 'left'")
                else:
                    key_down('right')
                    print("[DEBUG] ClimbRope: Key down 'right'")

                time.sleep(0.1)
                key_up('left')
                key_up('right')
                print("[DEBUG] ClimbRope: Key up 'left' and 'right'")

                # 更新 d_x
                d_x = self.rope_start[0] - config.player_pos[0]
                print(f"[DEBUG] ClimbRope: d_x after adjustment = {d_x}")

                # 檢查是否符合退出條件
                if not config.enabled or abs(d_x) <= self.jump_rope_tolerance:
                    break
            # 爬的動作
            print("[DEBUG] ClimbRope: Reached rope_start, jumping")
            press(Key.JUMP, 1)
            print("[DEBUG] ClimbRope: Jumped")
            key_down('left')
            time.sleep(0.2)
            key_up('left')
            time.sleep(0.2)
            key_down('right')
            time.sleep(0.2)
            key_up('right')
            time.sleep(0.5)

        #如果y到達目標高度(<settings.adjust_tolerance)，則判斷爬完沒：先左右走假如p[x]沒變則繼續按上
        d = 0
        curr_x = config.player_pos[0]
        while config.enabled and abs(curr_x - config.player_pos[0])<=settings.adjust_tolerance:
            if (d // 2) % 2 == 0:
                key_down('left')
                time.sleep(0.5)
                key_up('left')
            else:
                key_down('right')
                time.sleep(0.5)
                key_up('right')
                d = d + 1
                key_down('up')
                time.sleep(1)
                key_up('up')
                curr_x = config.player_pos[0]
                print(f"[DEBUG] ClimbRope: curr_x = {curr_x}, config.player_pos[0] = {config.player_pos[0]}")
    
            key_up('left')
            key_up('right')
        key_up('up')
        print("[DEBUG] ClimbRope: Key up 'left', 'right', and 'up'")

class JumpMove(Command):
    """跳，為了撿東西"""

    def __init__(self, jump_count=1):
        super().__init__(locals())
        self.jump_count = int(jump_count)

    def main(self):
        press(Key.JUMP, self.jump_count)

class TwinStarAttack(Command):
    """Uses 'Twin Star Attack' once."""
    def __init__(self, backswing=0.3):
        super().__init__(locals())
        self.backswing = float(backswing)


    def main(self):
        press(Key.TWIN_STAR_ATTACK, 3)
        time.sleep(self.backswing)

    def main(self):
        press(Key.TWIN_STAR_ATTACK, 3)
        time.sleep(self.backswing)
