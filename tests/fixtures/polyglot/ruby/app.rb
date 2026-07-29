require_relative "lib"

def process(x)
  helper(x)
end

def run
  Engine.new.process(process(1))
end
