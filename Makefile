DEMO_OUTPUTS = output/bayer_mosaick.png

all: clean $(DEMO_OUTPUTS)

output:
	mkdir -p output

$(DEMO_OUTPUTS): output
	./demo

clean:
	rm -rf output
