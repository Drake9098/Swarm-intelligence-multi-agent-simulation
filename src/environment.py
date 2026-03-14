import json


EMPTY = 0
WALL = 1
WAREHOUSE = 2
ENTRANCE = 3
EXIT = 4


class Environment:
    def __init__(self, json_path: str):
        """
        Inizializza l'ambiente di simulazione caricando i dati da un file JSON.

        Args:
            json_path (str): Il percorso del file JSON contenente i dati dell'ambiente.
        """

        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.grid = data["grid"]
        self.size = data["metadata"]["grid_size"]
        self.warehouses = data["warehouses"]

        self._objects = {i: tuple(pos) for i, pos in enumerate(data["objects"])}
        self._pos_to_obj: dict[tuple, int] = {pos: i for i, pos in self._objects.items()}
        self._delivered = set()
        self._claimed = set()   # oggetti attualmente portati da un agente (esclusi da reveal)

        # Normalizza entrance/exit a tuple per confronti coerenti (nel JSON sono liste)
        for w in self.warehouses:
            w["entrance"] = tuple(w["entrance"])
            w["exit"] = tuple(w["exit"])
    

    def in_bound(self, r: int, c: int) -> bool:
        """
        Verifica se una posizione (riga, colonna) è all'interno dei confini dell'ambiente.

        Args:
            r (int): La riga da verificare.
            c (int): La colonna da verificare.

        Returns:
            bool: True se la posizione è all'interno dei confini, False altrimenti.
        """
        return 0 <= r < self.size and 0 <= c < self.size
    

    def is_walkable(self, r: int, c: int, from_r: int, from_c: int) -> bool:
        """
        Verifica se una posizione (riga, colonna) è percorribile dall'agente che si trova in (from_r, from_c).

        Args:
            r (int): La riga della cella destinazione.
            c (int): La colonna della cella destinazione.
            from_r (int): La riga della posizione attuale dell'agente.
            from_c (int): La colonna della posizione attuale dell'agente.

        Returns:
            bool: True se la mossa è consentita, False altrimenti.
        """
        
        if not self.in_bound(r,c):
            return False
        
        cell_type = self.grid[r][c]
        if cell_type == WALL:
            return False
        elif cell_type == EMPTY or cell_type == WAREHOUSE:
            return True
        elif cell_type == ENTRANCE:
            for w in self.warehouses:
                if w["entrance"] == (r, c):
                    side = w["side"]
                    if side == "top" and from_r == r + 1 and from_c == c:
                        return True
                    elif side == "bottom" and from_r == r - 1 and from_c == c:
                        return True
                    elif side == "left" and from_c == c + 1 and from_r == r:
                        return True
                    elif side == "right" and from_c == c - 1 and from_r == r:
                        return True
            return False
        elif cell_type == EXIT:
            for w in self.warehouses:
                if w["exit"] == (r, c):
                    side = w["side"]
                    if side == "top" and from_r == r - 1 and from_c == c:
                        return True
                    elif side == "bottom" and from_r == r + 1 and from_c == c:
                        return True
                    elif side == "left" and from_c == c - 1 and from_r == r:
                        return True
                    elif side == "right" and from_c == c + 1 and from_r == r:
                        return True
            return False            

    
    def reveal_object_at(self, r: int, c: int) -> dict | None:
        """
        Rende visibile la posizione di un oggetto specifico.
        Restituisce None se l'oggetto è già stato raccolto (claimed) o consegnato.
        """
        obj_id = self._pos_to_obj.get((r, c))
        if obj_id is not None:
            return {"id": obj_id, "pos": (r, c)}
        return None
    

    def claim_object(self, obj_id: int) -> bool:
        """Prenota un oggetto per un agente (pickup esclusivo).
        Restituisce True se il claim ha avuto successo, False se già claimed da altri."""
        if obj_id in self._objects and obj_id not in self._claimed:
            self._pos_to_obj.pop(self._objects[obj_id], None)
            self._claimed.add(obj_id)
            return True
        return False


    def deliver_object(self, obj_id: int) -> bool:
        """Segna un oggetto come consegnato (rimuove da _objects e _claimed)."""
        if obj_id in self._objects:
            self._claimed.discard(obj_id)
            self._delivered.add(obj_id)
            del self._objects[obj_id]
            return True
        return False
    

    def get_warehouse_entrances(self) -> list:
        """
        Restituisce una lista di tuple (riga, colonna) per ogni entrata dei magazzini.

        Returns:
            list: Una lista di tuple (riga, colonna) per ogni entrata dei magazzini. 
        """

        return self.warehouses


    def objects_remaining(self) -> int:
        """Ritorna il numero di oggetti non ancora consegnati."""
        return len(self._objects)


    def all_delivered(self) -> bool:
        """Ritorna True se tutti gli oggetti sono stati consegnati."""
        return len(self._objects) == 0
