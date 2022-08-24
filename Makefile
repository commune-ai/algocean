down:
	./start.sh --purge
stop:
	./start.sh --purge
start:
	./start.sh 
up:
	./start.sh 
backend: 
	./start.sh --backend; docker exec -it ocean_backend_1;

restart:
	./start.sh --purge && ./start.sh;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ${arg} bash
build_backend:
	docker-compose -f "backend/backend.yml" build;

app:
	docker exec -it ocean_backend_1 bash -c "streamlit run ${arg}"

kill_all:
	docker kill $(docker ps -q)

logs:
	docker logs ${arg} --tail=100 --follow


enter_backend:
	docker exec -it ocean_backend_1 bash
