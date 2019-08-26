from asciimatics.screen import ManagedScreen
from asciimatics.effects import Print
from asciimatics.scene import Scene
from asciimatics.renderers import StaticRenderer, Box
from time import sleep

def RenderScene(screen, positions_dict):
    snake_renderer = StaticRenderer(images=[r'${6}X'])
    apple_renderer = StaticRenderer(images=[r'${7}O'])
    width, height = positions_dict['shape']
    snake_pos = positions_dict['snake']
    apple_pos = positions_dict['apple']
    effects = [Print(screen,Box(width+2, height+2),x=0,y=0,speed=0)]
    effects.append(Print(screen,apple_renderer,
                         x=apple_pos[0]+1,
                         y=apple_pos[1]+1,
                         speed=0))
    for segment in snake_pos:
        effects.append(Print(screen,snake_renderer,
                             x=segment[0]+1,
                             y=segment[1]+1,
                             speed=0))
    screen.set_scenes([Scene(effects)])
    screen.draw_next_frame(repeat=False)


@ManagedScreen
def ascii_snake(env,model,screen=None):
    obs, done = env.reset(), False
    obs_dict = env.render('dict')
    RenderScene(screen,obs_dict)
    while not done:
        obs, _, done, obs_dict = env.step(model(obs[None])[0])
        RenderScene(screen,obs_dict)
        sleep(0.1)
    print("Episode score:", obs_dict['score'])
