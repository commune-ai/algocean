down:
	./start.sh --purge
start:
	./start.sh 


restart:
	docker kill $(docker ps --filter "name=ocean" -q); ./start.sh;


prune_volumes:	
	docker system prune --all --volumes

bash:
	docker exec -it ${arg} bash
build_backend:
	docker-compose -f "backend/backend.yml" build;

app:
	docker exec -it ocean_backend_1 bash -c "streamlit run algocean/ocean/module.py"

kill_all:
	docker kill $(docker ps -q)