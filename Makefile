# Transform the machine arch into some standard values: "arm", "arm64", or "x86"
SYSTEM_ARCH := $(shell uname -m | sed -e 's/aarch64.*/arm64/' -e 's/x86_64.*/x86/' -e 's/armv.*/arm/')

# To build for an arch different from the current system, set this env var to 1 of the values in the comment above
ARCH ?= $(SYSTEM_ARCH)

# These variables can be overridden from the environment
VERSION ?= 0.3.4
DOCKER_NAME ?= aural2_${SYSTEM_ARCH}
DOCKER_HUB_ID ?= openhorizon

target/dockerimage_$(ARCH): Dockerfile.$(ARCH) webgui/templates/index.html webgui/templates/tag.html webgui/templates/vocab.html webgui/static/style.css gen_train_graph.py main.go vsh.go
	docker build -t $(DOCKER_NAME):$(VERSION) -f Dockerfile.$(ARCH) .
	touch target/dockerimage_$(ARCH)

clean:
	rm target/*

clean-dockerimages: clean
	docker rmi $(REG_PATH)/aural2:latest

run: target/dockerimage_$(ARCH)
	-docker kill aural2
	-mkdir /tmp/aural2 &
	-touch /tmp/aural2/label_store.db
	docker run -it --rm --name aural2 --net=microphone -p 48125:48125 --net-alias=aural2 -v /tmp/aural2:/persist $(DOCKER_NAME):$(VERSION)

gopherjs_loop:
	while inotifywait -e close_write webgui/main.go; do gopherjs build -o webgui/static/main.js webgui/main.go; done;

gopherjs:
	gopherjs build -o webgui/static/main.js webgui/main.go


# To publish you must have write access to the docker hub openhorizon user
publish: target/dockerimage_$(ARCH)
	docker tag $(DOCKER_NAME):$(VERSION) $(DOCKER_HUB_ID)/$(DOCKER_NAME):$(VERSION)
	docker push $(DOCKER_HUB_ID)/$(DOCKER_NAME):$(VERSION)
