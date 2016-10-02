build:
	moonc convnetmoon/*.moon
test:
	moonc convnetmoon/*.moon
	moonc test.moon
	lua test.lua
