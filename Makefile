build:
	moonc convnetmoon/*.moon
	moonc convnetmoon/*/*.moon
test:
	moonc convnetmoon/*.moon
	moonc convnetmoon/*/*.moon

	moonc test.moon
	lua test.lua
