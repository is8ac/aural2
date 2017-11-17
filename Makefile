SYSTEM_ARCH := $(shell uname -m | sed 's/\(...\).*/\1/')
VERSION=1.0.1

REGISTRY=summit.hovitos.engineering
ifeq ($(SYSTEM_ARCH),arm)
REG_PATH := $(REGISTRY)/armhf
else
REG_PATH := $(REGISTRY)/x86
endif

GROUP_ID := $(shell id -g)
USER_ID := $(shell id -u)

target/dockerimage: target/aural2 target/train_graph.pb target/main.js Dockerfile.$(SYSTEM_ARCH) webgui/templates/index.html webgui/templates/tag.html webgui/templates/vocab.html webgui/static/style.css run.sh
	docker build -t $(REG_PATH)/aural2 -f Dockerfile.$(SYSTEM_ARCH) .
	touch target/dockerimage

target/aural2 target/main.js target/train_graph.pb: target/builddockerimage blob_compute.go  http.go  main.go  train.go  vsh.go webgui/main.go gen_train_graph.py
	docker run -it \
	-v $$GOPATH/src/github.ibm.com/Blue-Horizon/aural2:/root/go/src/github.ibm.com/Blue-Horizon/aural2:ro \
	-v $$GOPATH/src/github.ibm.com/Blue-Horizon/aural2/target:/root/go/src/github.ibm.com/Blue-Horizon/aural2/target \
	--workdir /root/go/src/github.ibm.com/Blue-Horizon/aural2 \
	aural2_build \
	sh -c "go build -o target/aural2 && /root/go/bin/gopherjs build -o target/main.js webgui/main.go && python gen_train_graph.py && chown $(USER_ID):$(GROUP_ID) target/*"

target/builddockerimage: Dockerfile_build.$(SYSTEM_ARCH)
	docker build -t aural2_build -f Dockerfile_build.$(SYSTEM_ARCH) .
	touch target/builddockerimage

dockerimagesquash: target/dockerimage
	docker export `docker run -d $(REG_PATH)/aural2 ls` | docker import --change "CMD /start.sh" - $(REG_PATH)/aural2:$(VERSION)

push: dockerimagesquash
	docker push $(REG_PATH)/aural2:$(VERSION)

clean:
	rm target/*

clean-dockerimages: clean
	docker rmi $(REG_PATH)/aural2:latest

run: target/dockerimage
	touch /tmp/aural2/label_store.db
	docker run -it -p 48125:48125 --privileged -v /tmp/aural2/audio:/audio -v /tmp/aural2/label_store.db:/label_store.db summit.hovitos.engineering/x86/aural2
