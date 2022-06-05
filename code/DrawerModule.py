from turtle import window_height, window_width
import pyglet

class Rect:
  def __init__(self, x, y, w, h):
    self.set(x, y, w, h)

  def draw(self, color=(255, 255, 255)):
    pyglet.graphics.draw(
      4,
      pyglet.gl.GL_QUADS,
      self._quad,
      ('c3B', color * 4)
    )

  def set(self, x=None, y=None, w=None, h=None):
    self._x = self._x if x is None else x
    self._y = self._y if y is None else y
    self._w = self._w if w is None else w
    self._h = self._h if h is None else h
    self._quad = ('v2f', (self._x, self._y,
                          self._x + self._w, self._y,
                          self._x + self._w, self._y + self._h,
                          self._x, self._y + self._h))

  def __repr__(self):
    return f"Rect(x={self._x}, y={self._y}, w={self._w}, h={self._h})"

class CustomLoop(pyglet.app.EventLoop):
  def idle(self):    
    dt = self.clock.update_time()
    self.clock.call_scheduled_functions(dt)

    # Redraw all windows
    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_event('on_draw')
        window.flip()
        window._legacy_invalid = False

    # no timout (sleep-time between idle()-calls)
    return 0

class Drawer:
    def __init__(self, env, model, rows=2, seconds_per_frame=0.5, fullscreen=True):
      from time import time

      self.prev_timestamp = time() - 2 * seconds_per_frame

      self.state = env.reset()
      self.done = False
      self.score = 0.0

      window = pyglet.window.Window(fullscreen=fullscreen)

      @window.event
      def on_draw():
        def drawRect(x, y, w, h, color):
          r = Rect(x, y, w, h)
          r.draw(color=color)

        FILL_QUANTILE = 0.7
        new_timestamp = time()
        if new_timestamp - self.prev_timestamp >= seconds_per_frame and not self.done:
          self.prev_timestamp = new_timestamp
          action, _ = model.predict(self.state)
          self.state, reward, self.done, additional_info = env.step(action)
          self.score += reward
        window.clear()
        label = pyglet.text.Label(
          text=f'Current score: {self.score}',
          x=window.width * 0.1,
          y=window.height * (1 + FILL_QUANTILE) / 2,
          font_size=28
        )
        label.draw()
        cols = (len(env.servers) + rows - 1) // rows
        max_mem = max(server._smem for server in env.servers)
        max_cpu = max(server._scpu for server in env.servers)
        max_h = FILL_QUANTILE * window.height / rows
        W = window.width / cols
        gray = (100, 100, 100)
        red = (219, 88, 86)
        blue = (93, 155, 155)
        for i in range(rows):
          for j in range(cols):
            id = i * cols + j
            if id < len(env.servers):
              server = env.servers[id]
              scpu = server._scpu / max_cpu
              smem = server._smem / max_mem
              cpu = (server._scpu - server.cpu) / max_cpu
              mem = (server._smem - server.mem) / max_mem
              lx = j / cols * window.width
              ly = i / rows * window.height * FILL_QUANTILE
              drawRect(lx, ly, W / 2, scpu * max_h, gray)
              drawRect(lx + W / 2, ly, W / 2, smem * max_h, gray)
              drawRect(lx, ly, W / 2, cpu * max_h, blue)
              drawRect(lx + W / 2, ly, W / 2, mem * max_h, red)

      pyglet.app.event_loop = CustomLoop()
      pyglet.app.run()