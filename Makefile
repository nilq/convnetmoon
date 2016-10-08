build:
	moonc convnetmoon/*.moon
	moonc convnetmoon/*/*.moon
test:
	moonc test.moon
	lua test.lua
qlearn:
	moonc deepqlearn.moon
	lua deepqlearn.lua
