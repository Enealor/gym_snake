from asciimatics.screen import ManagedScreen
from asciimatics.effects import Print
from asciimatics.scene import Scene
from asciimatics.renderers import StaticRenderer, Box
from time import sleep

def RenderScene(screen, positions_dict):
    snake_renderer = StaticRenderer(images=[r'${6}X'])
    apple_renderer = StaticRenderer(images=[r'${7}O'])

    #Establish grid
    width, height = positions_dict['shape']
    effects = [Print(screen,Box(width+2, height+2),x=0,y=0,speed=0)]
    #Add apple
    apple_x,apple_y = positions_dict['apple']
    effects.append(Print(screen,apple_renderer,
                         x=apple_x+1,
                         y=apple_y+1,
                         speed=0))
    #Add snake
    snake_pos = positions_dict['snake']
    for segment in snake_pos:
        effects.append(Print(screen,snake_renderer,
                             x=segment[0]+1,
                             y=segment[1]+1,
                             speed=0))
    #Adds and draws next frame.
    screen.set_scenes([Scene(effects)])
    screen.draw_next_frame(repeat=False)


@ManagedScreen
def ascii_snake(env,model,screen=None):
    #Set up environment
    obs, done = env.reset(), False
    info = env.render('dict')
    #Render first scene
    RenderScene(screen,info)
    while not done:
        #Predict best move
        action, _states = model.predict(obs)
        #Use that move
        obs, rewards, done, _ = env.step(action)
        info = env.render('dict')
        #Render new screen
        RenderScene(screen,info)
        sleep(0.1)
    print("Episode score:", info['score'])
