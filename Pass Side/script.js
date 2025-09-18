// Variables globales
let gameSettings = {
    soundEffects: true,
    theme: 'dark', // 'dark' o 'light'
    playerName: 'Jugador',
    cpuDifficulty: 'beginner', // 'beginner', 'intermediate', 'expert'
    boardSize: 'classic' // 'express', 'classic', 'master'
};

// Sistema de distribuciones ponderadas
const distributions = [
    { name: "Centro Puro", pattern: "xooooooox", weight: 0.20 },
    { name: "Lateral Derecho", pattern: "xxooooooo", weight: 0.10 },
    { name: "Lateral Izquierdo", pattern: "oooooooxx", weight: 0.10 },
    { name: "Alternada Compacta", pattern: "oxoooooxo", weight: 0.10 },
    { name: "Centro con Flancos", pattern: "oooxoxooo", weight: 0.10 },
    { name: "Triple Centro", pattern: "ooxoooxoo", weight: 0.10 },
    { name: "Distribución Aleatoria", pattern: "random", weight: 0.30 }
];

// Patrones específicos para cada tamaño de tablero
const BOARD_PATTERNS = {
    express: [
        { name: "Centro Puro", pattern: "oxoxo", weight: 0.30 }, // 3 fichas: posiciones 0, 2, 4
        { name: "Lateral", pattern: "oxoox", weight: 0.20 },     // 3 fichas: posiciones 0, 2, 3
        { name: "Compacta", pattern: "ooxox", weight: 0.25 },    // 3 fichas: posiciones 0, 1, 3
        { name: "Distribución Aleatoria", pattern: "random", weight: 0.25 }
    ],
    classic: [
        { name: "Centro Puro", pattern: "xooooooox", weight: 0.20 },
        { name: "Lateral Derecho", pattern: "xxooooooo", weight: 0.10 },
        { name: "Lateral Izquierdo", pattern: "oooooooxx", weight: 0.10 },
        { name: "Alternada Compacta", pattern: "oxoooooxo", weight: 0.10 },
        { name: "Centro con Flancos", pattern: "oooxoxooo", weight: 0.10 },
        { name: "Triple Centro", pattern: "ooxoooxoo", weight: 0.10 },
        { name: "Distribución Aleatoria", pattern: "random", weight: 0.30 }
    ],
    master: [
        { name: "Doble Núcleo", pattern: "xooooxoooox", weight: 0.20 }, // 8 fichas
        { name: "Fortaleza Lateral", pattern: "ooooxxxoooo", weight: 0.10 }, // 8 fichas  
        { name: "Cuatro Carriles", pattern: "ooxooxooxoo", weight: 0.10 }, // 8 fichas
        { name: "Cadena Alterna", pattern: "oxoooxoooxo", weight: 0.10 }, // 8 fichas
        { name: "Lateral Derecho", pattern: "xxxoooooooo", weight: 0.05 }, // 8 fichas
        { name: "Lateral Izquierdo", pattern: "ooooooooxxx", weight: 0.05 }, // 8 fichas
        { name: "Control Lateral", pattern: "oooxoxoxooo", weight: 0.10 }, // 8 fichas
        { name: "Distribución Aleatoria", pattern: "random", weight: 0.30 }
    ]
};

// Función para formatear probabilidades
function formatProbability(weight) {
    const percentage = Math.round(weight * 100);
    return percentage < 1 ? '<1%' : `${percentage}%`;
}

// Función para obtener distribución ponderada
function getWeightedDistribution() {
    const patterns = BOARD_PATTERNS[gameSettings.boardSize];
    const random = Math.random();
    
    // Verificar si hay distribución aleatoria
    const randomDist = patterns.find(d => d.pattern === "random");
    if (randomDist && random < randomDist.weight) {
        return generateRandomDistribution();
    }
    
    // Usar distribuciones predefinidas
    const predefinedDistributions = patterns.filter(d => d.pattern !== "random");
    const totalWeight = predefinedDistributions.reduce((sum, dist) => sum + dist.weight, 0);
    let weightedRandom = random * totalWeight;
    
    for (let dist of predefinedDistributions) {
        weightedRandom -= dist.weight;
        if (weightedRandom <= 0) {
            return dist;
        }
    }
    
    return predefinedDistributions[0]; // Fallback
}

// Función para generar distribución completamente aleatoria
function generateRandomDistribution() {
    const config = BOARD_CONFIGS[gameSettings.boardSize];
    const numPieces = config.pieces;
    const numCols = config.cols;
    
    // Calcular el número total de combinaciones posibles
    const totalCombinations = factorial(numCols) / (factorial(numPieces) * factorial(numCols - numPieces));
    
    // Calcular el peso individual de cada distribución aleatoria
    const patterns = BOARD_PATTERNS[gameSettings.boardSize];
    const randomDist = patterns.find(d => d.pattern === "random");
    const individualWeight = randomDist ? randomDist.weight / totalCombinations : 0.01;
    
    const positions = Array.from({length: numCols}, (_, i) => i);
    const selectedPositions = [];
    
    // Seleccionar el número correcto de posiciones aleatorias
    while (selectedPositions.length < numPieces) {
        const randomIndex = Math.floor(Math.random() * positions.length);
        const selectedPosition = positions.splice(randomIndex, 1)[0];
        selectedPositions.push(selectedPosition);
    }
    
    // Crear el patrón
    let pattern = "x".repeat(numCols); // Espacios vacíos
    for (let pos of selectedPositions) {
        pattern = pattern.substring(0, pos) + "o" + pattern.substring(pos + 1);
    }
    
    return {
        name: "Aleatoria",
        pattern: pattern,
        weight: individualWeight
    };
}

// Función auxiliar para calcular factorial
function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Sistema de audio
const audioManager = {
    // Contexto de audio
    audioContext: null,
    
    // Inicializar el sistema de audio
    init() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Verificar que los archivos de trompeta se cargan correctamente
            this.checkAudioFiles();
        } catch (e) {
            console.log('AudioContext no soportado:', e);
        }
    },
    
    // Verificar que los archivos de audio se cargan
    checkAudioFiles() {
        const trumpetVictory = document.getElementById('trumpetVictory');
        const trumpetDefeat = document.getElementById('trumpetDefeat');
        
        if (trumpetVictory) {
            trumpetVictory.addEventListener('loadeddata', () => {
                console.log('Archivo de trompeta de victoria cargado correctamente');
            });
            trumpetVictory.addEventListener('error', (e) => {
                console.log('Error cargando trompeta de victoria:', e);
            });
        }
        
        if (trumpetDefeat) {
            trumpetDefeat.addEventListener('loadeddata', () => {
                console.log('Archivo de trompeta de derrota cargado correctamente');
            });
            trumpetDefeat.addEventListener('error', (e) => {
                console.log('Error cargando trompeta de derrota:', e);
            });
        }
    },
    
    // Reproducir sonido de clic en botón
    playButtonClick() {
        if (gameSettings.soundEffects && this.audioContext) {
            try {
                // Crear un oscilador para el sonido de clic
                const oscillator = this.audioContext.createOscillator();
                const gainNode = this.audioContext.createGain();
                
                // Configurar el sonido
                oscillator.type = 'sine';
                oscillator.frequency.setValueAtTime(800, this.audioContext.currentTime); // Frecuencia inicial
                oscillator.frequency.exponentialRampToValueAtTime(400, this.audioContext.currentTime + 0.1); // Decaimiento
                
                // Configurar el volumen
                gainNode.gain.setValueAtTime(0.2, this.audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.1);
                
                // Conectar los nodos
                oscillator.connect(gainNode);
                gainNode.connect(this.audioContext.destination);
                
                // Reproducir el sonido
                oscillator.start(this.audioContext.currentTime);
                oscillator.stop(this.audioContext.currentTime + 0.1);
            } catch (e) {
                console.log('No se pudo reproducir el sonido:', e);
            }
        }
    },
    
    // Reproducir sonido de movimiento de ficha
    playPieceMove() {
        if (gameSettings.soundEffects && this.audioContext) {
            try {
                // Crear dos osciladores para simular el sonido de ficha en tablero
                const oscillator1 = this.audioContext.createOscillator();
                const oscillator2 = this.audioContext.createOscillator();
                const gainNode = this.audioContext.createGain();
                const filter = this.audioContext.createBiquadFilter();
                
                // Configurar el primer oscilador (sonido principal)
                oscillator1.type = 'sine';
                oscillator1.frequency.setValueAtTime(200, this.audioContext.currentTime);
                oscillator1.frequency.exponentialRampToValueAtTime(150, this.audioContext.currentTime + 0.05);
                
                // Configurar el segundo oscilador (armónico)
                oscillator2.type = 'sine';
                oscillator2.frequency.setValueAtTime(400, this.audioContext.currentTime);
                oscillator2.frequency.exponentialRampToValueAtTime(300, this.audioContext.currentTime + 0.05);
                
                // Configurar el filtro para simular el sonido de madera
                filter.type = 'lowpass';
                filter.frequency.setValueAtTime(800, this.audioContext.currentTime);
                filter.Q.setValueAtTime(1, this.audioContext.currentTime);
                
                // Configurar el volumen con decaimiento natural
                gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.2);
                
                // Conectar los nodos
                oscillator1.connect(gainNode);
                oscillator2.connect(gainNode);
                gainNode.connect(filter);
                filter.connect(this.audioContext.destination);
                
                // Reproducir el sonido
                oscillator1.start(this.audioContext.currentTime);
                oscillator2.start(this.audioContext.currentTime);
                oscillator1.stop(this.audioContext.currentTime + 0.2);
                oscillator2.stop(this.audioContext.currentTime + 0.2);
            } catch (e) {
                console.log('No se pudo reproducir el sonido de movimiento:', e);
            }
        }
    },
    
    // Reproducir sonido de celebración al llegar a la meta
    playGoalCelebration() {
        if (gameSettings.soundEffects && this.audioContext) {
            try {
                // Crear una secuencia de notas ascendentes para celebrar
                const notes = [523.25, 659.25, 783.99]; // C5, E5, G5 (acorde mayor)
                const noteDuration = 0.15;
                const totalDuration = notes.length * noteDuration;
                
                notes.forEach((frequency, index) => {
                    const oscillator = this.audioContext.createOscillator();
                    const gainNode = this.audioContext.createGain();
                    
                    // Configurar el oscilador
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime + index * noteDuration);
                    
                    // Configurar el volumen con ataque y decaimiento
                    gainNode.gain.setValueAtTime(0, this.audioContext.currentTime + index * noteDuration);
                    gainNode.gain.linearRampToValueAtTime(0.2, this.audioContext.currentTime + index * noteDuration + 0.02);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + index * noteDuration + noteDuration);
                    
                    // Conectar los nodos
                    oscillator.connect(gainNode);
                    gainNode.connect(this.audioContext.destination);
                    
                    // Reproducir la nota
                    oscillator.start(this.audioContext.currentTime + index * noteDuration);
                    oscillator.stop(this.audioContext.currentTime + index * noteDuration + noteDuration);
                });
            } catch (e) {
                console.log('No se pudo reproducir el sonido de celebración:', e);
            }
        }
    },
    
    // Reproducir sonido antagonista cuando el rival llega a la meta
    playRivalGoalWarning() {
        if (gameSettings.soundEffects && this.audioContext) {
            try {
                // Crear una secuencia de notas descendentes y disonantes para generar preocupación
                const notes = [440, 392, 349.23]; // A4, G4, F4 (descendente, más grave)
                const noteDuration = 0.2;
                const totalDuration = notes.length * noteDuration;
                
                notes.forEach((frequency, index) => {
                    const oscillator = this.audioContext.createOscillator();
                    const gainNode = this.audioContext.createGain();
                    
                    // Configurar el oscilador con onda cuadrada para sonido más duro
                    oscillator.type = 'square';
                    oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime + index * noteDuration);
                    
                    // Configurar el volumen con ataque rápido y decaimiento lento
                    gainNode.gain.setValueAtTime(0, this.audioContext.currentTime + index * noteDuration);
                    gainNode.gain.linearRampToValueAtTime(0.15, this.audioContext.currentTime + index * noteDuration + 0.01);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + index * noteDuration + noteDuration);
                    
                    // Conectar los nodos
                    oscillator.connect(gainNode);
                    gainNode.connect(this.audioContext.destination);
                    
                    // Reproducir la nota
                    oscillator.start(this.audioContext.currentTime + index * noteDuration);
                    oscillator.stop(this.audioContext.currentTime + index * noteDuration + noteDuration);
                });
            } catch (e) {
                console.log('No se pudo reproducir el sonido de advertencia rival:', e);
            }
        }
    },
    
    // Reproducir sonido de eliminación
    playElimination(isPlayerEliminating) {
        if (gameSettings.soundEffects && this.audioContext) {
            try {
                if (isPlayerEliminating) {
                    // Sonido satisfactorio cuando el jugador elimina
                    this.playSatisfyingElimination();
                } else {
                    // Sonido frustrante cuando eliminan al jugador
                    this.playFrustratingElimination();
                }
            } catch (e) {
                console.log('No se pudo reproducir el sonido de eliminación:', e);
            }
        }
    },
    
    // Sonido satisfactorio para cuando eliminas
    playSatisfyingElimination() {
        // Sonido de "impacto" satisfactorio con resonancia
        const oscillator1 = this.audioContext.createOscillator();
        const oscillator2 = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        const filter = this.audioContext.createBiquadFilter();
        
        // Frecuencias que suenan "victoriosas"
        oscillator1.type = 'sine';
        oscillator1.frequency.setValueAtTime(400, this.audioContext.currentTime);
        oscillator1.frequency.exponentialRampToValueAtTime(300, this.audioContext.currentTime + 0.1);
        
        oscillator2.type = 'sine';
        oscillator2.frequency.setValueAtTime(600, this.audioContext.currentTime);
        oscillator2.frequency.exponentialRampToValueAtTime(450, this.audioContext.currentTime + 0.1);
        
        // Filtro para dar resonancia
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(1000, this.audioContext.currentTime);
        filter.Q.setValueAtTime(2, this.audioContext.currentTime);
        
        // Volumen con ataque rápido y decaimiento lento
        gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.25, this.audioContext.currentTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.3);
        
        // Conectar
        oscillator1.connect(gainNode);
        oscillator2.connect(gainNode);
        gainNode.connect(filter);
        filter.connect(this.audioContext.destination);
        
        // Reproducir
        oscillator1.start(this.audioContext.currentTime);
        oscillator2.start(this.audioContext.currentTime);
        oscillator1.stop(this.audioContext.currentTime + 0.3);
        oscillator2.stop(this.audioContext.currentTime + 0.3);
    },
    
    // Sonido frustrante para cuando te eliminan
    playFrustratingElimination() {
        // Sonido disonante y abrupto
        const oscillator1 = this.audioContext.createOscillator();
        const oscillator2 = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        const filter = this.audioContext.createBiquadFilter();
        
        // Frecuencias disonantes que suenan "mal"
        oscillator1.type = 'sawtooth';
        oscillator1.frequency.setValueAtTime(200, this.audioContext.currentTime);
        oscillator1.frequency.exponentialRampToValueAtTime(150, this.audioContext.currentTime + 0.08);
        
        oscillator2.type = 'sawtooth';
        oscillator2.frequency.setValueAtTime(250, this.audioContext.currentTime); // Disonancia
        oscillator2.frequency.exponentialRampToValueAtTime(180, this.audioContext.currentTime + 0.08);
        
        // Filtro para hacer el sonido más "áspero"
        filter.type = 'highpass';
        filter.frequency.setValueAtTime(300, this.audioContext.currentTime);
        filter.Q.setValueAtTime(3, this.audioContext.currentTime);
        
        // Volumen con ataque abrupto y decaimiento rápido
        gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.3, this.audioContext.currentTime + 0.005);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.15);
        
        // Conectar
        oscillator1.connect(gainNode);
        oscillator2.connect(gainNode);
        gainNode.connect(filter);
        filter.connect(this.audioContext.destination);
        
        // Reproducir
        oscillator1.start(this.audioContext.currentTime);
        oscillator2.start(this.audioContext.currentTime);
        oscillator1.stop(this.audioContext.currentTime + 0.15);
        oscillator2.stop(this.audioContext.currentTime + 0.15);
    },
    
    // Sonido de victoria del jugador
    playVictory() {
        console.log('Intentando reproducir sonido de victoria...');
        if (gameSettings.soundEffects) {
            const trumpetVictory = document.getElementById('trumpetVictory');
            console.log('Elemento de trompeta encontrado:', trumpetVictory);
            if (trumpetVictory) {
                trumpetVictory.currentTime = 0;
                console.log('Reproduciendo trompeta de victoria...');
                trumpetVictory.play().then(() => {
                    console.log('Trompeta de victoria reproducida exitosamente');
                }).catch(e => {
                    console.log('No se pudo reproducir el sonido de trompeta de victoria:', e);
                    this.playVictoryFallback();
                });
            } else {
                console.log('Elemento de trompeta no encontrado, usando fallback');
                this.playVictoryFallback();
            }
        } else {
            console.log('Efectos de sonido desactivados');
        }
    },
    
    // Sonido de victoria de respaldo (generado)
    playVictoryFallback() {
        if (this.audioContext) {
            try {
                // Melodía de victoria más elaborada y larga
                const victoryMelody = [
                    { freq: 523.25, duration: 0.3 }, // C5
                    { freq: 659.25, duration: 0.3 }, // E5
                    { freq: 783.99, duration: 0.3 }, // G5
                    { freq: 1046.50, duration: 0.4 }, // C6
                    { freq: 783.99, duration: 0.2 }, // G5
                    { freq: 1046.50, duration: 0.6 }  // C6 (final)
                ];
                
                let currentTime = this.audioContext.currentTime;
                
                victoryMelody.forEach((note, index) => {
                    const oscillator = this.audioContext.createOscillator();
                    const gainNode = this.audioContext.createGain();
                    const filter = this.audioContext.createBiquadFilter();
                    
                    // Configurar oscilador
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(note.freq, currentTime);
                    
                    // Filtro para suavizar el sonido
                    filter.type = 'lowpass';
                    filter.frequency.setValueAtTime(1200, currentTime);
                    filter.Q.setValueAtTime(1, currentTime);
                    
                    // Volumen con ataque suave y decaimiento
                    gainNode.gain.setValueAtTime(0, currentTime);
                    gainNode.gain.linearRampToValueAtTime(0.25, currentTime + 0.05);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, currentTime + note.duration);
                    
                    // Conectar
                    oscillator.connect(gainNode);
                    gainNode.connect(filter);
                    filter.connect(this.audioContext.destination);
                    
                    // Reproducir
                    oscillator.start(currentTime);
                    oscillator.stop(currentTime + note.duration);
                    
                    currentTime += note.duration;
                });
            } catch (e) {
                console.log('No se pudo reproducir el sonido de victoria:', e);
            }
        }
    },
    
    // Sonido de derrota del jugador
    playDefeat() {
        console.log('Intentando reproducir sonido de derrota...');
        if (gameSettings.soundEffects) {
            const trumpetDefeat = document.getElementById('trumpetDefeat');
            console.log('Elemento de trompeta de derrota encontrado:', trumpetDefeat);
            if (trumpetDefeat) {
                trumpetDefeat.currentTime = 0;
                console.log('Reproduciendo trompeta de derrota...');
                trumpetDefeat.play().then(() => {
                    console.log('Trompeta de derrota reproducida exitosamente');
                }).catch(e => {
                    console.log('No se pudo reproducir el sonido de trompeta de derrota:', e);
                    this.playDefeatFallback();
                });
            } else {
                console.log('Elemento de trompeta de derrota no encontrado, usando fallback');
                this.playDefeatFallback();
            }
        } else {
            console.log('Efectos de sonido desactivados');
        }
    },
    
    // Sonido de derrota de respaldo (generado)
    playDefeatFallback() {
        if (this.audioContext) {
            try {
                // Melodía de derrota descendente y melancólica
                const defeatMelody = [
                    { freq: 523.25, duration: 0.4 }, // C5
                    { freq: 466.16, duration: 0.4 }, // A#4
                    { freq: 392.00, duration: 0.4 }, // G4
                    { freq: 349.23, duration: 0.6 }, // F4
                    { freq: 311.13, duration: 0.8 }  // D#4 (final grave)
                ];
                
                let currentTime = this.audioContext.currentTime;
                
                defeatMelody.forEach((note, index) => {
                    const oscillator = this.audioContext.createOscillator();
                    const gainNode = this.audioContext.createGain();
                    const filter = this.audioContext.createBiquadFilter();
                    
                    // Configurar oscilador
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(note.freq, currentTime);
                    
                    // Filtro para hacer el sonido más grave y melancólico
                    filter.type = 'lowpass';
                    filter.frequency.setValueAtTime(800, currentTime);
                    filter.Q.setValueAtTime(2, currentTime);
                    
                    // Volumen con ataque suave y decaimiento lento
                    gainNode.gain.setValueAtTime(0, currentTime);
                    gainNode.gain.linearRampToValueAtTime(0.2, currentTime + 0.1);
                    gainNode.gain.exponentialRampToValueAtTime(0.01, currentTime + note.duration);
                    
                    // Conectar
                    oscillator.connect(gainNode);
                    gainNode.connect(filter);
                    filter.connect(this.audioContext.destination);
                    
                    // Reproducir
                    oscillator.start(currentTime);
                    oscillator.stop(currentTime + note.duration);
                    
                    currentTime += note.duration;
                });
            } catch (e) {
                console.log('No se pudo reproducir el sonido de derrota:', e);
            }
        }
    }
};

// Variables del juego
let gameState = {
    board: [],
    currentPlayer: 'blue', // 'red' o 'blue'
    redPieces: 9,
    bluePieces: 9,
    selectedPiece: null,
    // Estadísticas de jugadores
    redEliminated: 0,
    blueEliminated: 0,
    redArrived: 0,
    blueArrived: 0,
    redPoints: 0,
    bluePoints: 0,
    // Contador de turno
    turnNumber: 1,
    // Sistema de tiempo
    gameStartTime: null,
    gameEndTime: null,
    gameDuration: 0,
    timeInterval: null,
    // Sistema de sugerencias
    showingHints: false,
    hintMoves: [],
    // Estado del juego
    gameEnded: false,
    // Distribuciones de la partida
    redDistribution: null,
    blueDistribution: null,
    // Estado de la animación inicial
    showingFormationAnimation: true
};

// Constantes del tablero (se actualizarán dinámicamente)
let BOARD_ROWS = 11;
let BOARD_COLS = 9;
let POINTS_TO_WIN = 7; // Se actualizará dinámicamente

// Configuraciones de tablero
const BOARD_CONFIGS = {
    express: { rows: 7, cols: 5, pieces: 3, points: 3 },
    classic: { rows: 11, cols: 9, pieces: 7, points: 7 },
    master: { rows: 15, cols: 11, pieces: 8, points: 8 }
};

// Función para configurar el tablero según el tamaño seleccionado
function configureBoard() {
    const config = BOARD_CONFIGS[gameSettings.boardSize];
    BOARD_ROWS = config.rows;
    BOARD_COLS = config.cols;
    POINTS_TO_WIN = config.points; // Proporción 1:1 (1 punto por ficha)
    gameState.redPieces = config.pieces;
    gameState.bluePieces = config.pieces;
    
    // Actualizar las filas de meta dinámicamente
    BLUE_GOAL_ROW = 0; // Siempre la primera fila
    RED_GOAL_ROW = BOARD_ROWS - 1; // Siempre la última fila
    RED_START_ROW = 1; // Siempre la segunda fila
    BLUE_START_ROW = BOARD_ROWS - 2; // Siempre la penúltima fila
    
    // Calcular y aplicar el tamaño de casilla apropiado
    updateBoardCellSize();
}

// Función para actualizar el tamaño de las casillas según el tablero
function updateBoardCellSize() {
    let targetWidth, currentWidth;
    
    // Configuración específica por tablero
    if (gameSettings.boardSize === 'express') {
        targetWidth = 375; // Ancho deseado para Express
        currentWidth = targetWidth / BOARD_COLS; // Tamaño de casilla para Express
    } else {
        // Para Clásico y Master, mantener el ancho original
        const classicWidth = 55 * 9; // Ancho del tablero clásico (55px * 9 columnas)
        currentWidth = classicWidth / BOARD_COLS;
    }
    
    // Las filas de meta se ajustan al ancho actual del tablero
    const goalRowHeight = 40; // Siempre 40px como en el modo clásico
    const goalRowWidth = currentWidth * BOARD_COLS; // Ancho igual al tablero actual
    
    // Calcular tamaño de las fichas
    let pieceSize;
    if (gameSettings.boardSize === 'express') {
        // Para Express, fichas del 80% del tamaño proporcional
        pieceSize = Math.round(currentWidth * 0.73 * 0.8);
    } else {
        // Para Clásico y Master, tamaño proporcional normal (73% del tamaño de casilla)
        pieceSize = Math.round(currentWidth * 0.73);
    }
    
    // Actualizar las variables CSS
    document.documentElement.style.setProperty('--cell-size', `${currentWidth}px`);
    document.documentElement.style.setProperty('--goal-row-width', `${goalRowWidth}px`);
    document.documentElement.style.setProperty('--goal-row-height', `${goalRowHeight}px`);
    document.documentElement.style.setProperty('--piece-size', `${pieceSize}px`);
}

// Variables dinámicas para las filas de meta
let BLUE_GOAL_ROW = 0; // Siempre la primera fila
let RED_GOAL_ROW = BOARD_ROWS - 1; // Siempre la última fila
let RED_START_ROW = 1; // Siempre la segunda fila
let BLUE_START_ROW = BOARD_ROWS - 2; // Siempre la penúltima fila

// Constantes del juego
const VICTORY_CHECK_THRESHOLD = 3; // Verificar victoria cuando estés a 3 puntos o menos

const CELL_TYPES = {
    BLUE_GOAL: 'blue-goal',    // Fila 1 (índice 0)
    RED_START: 'red-start',    // Fila 2 (índice 1)
    NEUTRAL: 'neutral',        // Filas 3-4 (índices 2-3)
    SAFE_ZONE: 'safe-zone',    // Fila 5 (índice 4) - zona segura
    NEUTRAL2: 'neutral2',      // Filas 6-7 (índices 5-6)
    BLUE_START: 'blue-start',  // Fila 9 (índice 8)
    RED_GOAL: 'red-goal'       // Fila 10 (índice 9)
};

// Funciones del tablero
function initializeBoard() {
    gameState.board = [];
    
    // Crear el tablero vacío
    for (let row = 0; row < BOARD_ROWS; row++) {
        gameState.board[row] = [];
        for (let col = 0; col < BOARD_COLS; col++) {
            gameState.board[row][col] = {
                type: getCellType(row),
                piece: null,
                row: row,
                col: col
            };
        }
    }
    
    // Colocar las fichas iniciales
    placePieces();
}

function getCellType(row) {
    const totalRows = BOARD_ROWS;
    const middleRow = Math.floor(totalRows / 2);
    
    if (row === BLUE_GOAL_ROW) return CELL_TYPES.BLUE_GOAL;    // Meta azul (siempre primera fila)
    if (row === totalRows - 1) return CELL_TYPES.RED_GOAL;    // Meta roja (siempre última fila)
    if (row === RED_START_ROW) return CELL_TYPES.RED_START;    // Inicio rojo (siempre segunda fila)
    if (row === totalRows - 2) return CELL_TYPES.BLUE_START;    // Inicio azul (siempre penúltima fila)
    if (row === middleRow) return CELL_TYPES.SAFE_ZONE;    // Zona segura (fila del medio)
    
    // Filas neutrales
    if (row < middleRow) {
        return CELL_TYPES.NEUTRAL;    // Campo del jugador azul
    } else {
        return CELL_TYPES.NEUTRAL2;   // Campo del jugador rojo
    }
}

function placePieces() {
    // Obtener distribución independiente para cada equipo
    gameState.redDistribution = getWeightedDistribution();
    gameState.blueDistribution = getWeightedDistribution();
    
    console.log('🎲 Distribución roja seleccionada:', gameState.redDistribution.name, gameState.redDistribution.pattern);
    console.log('🎲 Distribución azul seleccionada:', gameState.blueDistribution.name, gameState.blueDistribution.pattern);
    
    const redStartRow = 1; // Fila de inicio rojo (siempre segunda fila)
    const blueStartRow = BOARD_ROWS - 2; // Fila de inicio azul (siempre penúltima fila)
    
    // Colocar fichas rojas en la fila de inicio rojo según su patrón
    let redPieceCount = 0;
    for (let col = 0; col < BOARD_COLS; col++) {
        if (gameState.redDistribution.pattern[col] === 'o') {
        gameState.board[redStartRow][col].piece = {
            color: 'red',
                id: `red_${redPieceCount}`
        };
            redPieceCount++;
        }
    }
    
    // Colocar fichas azules en la fila de inicio azul según su patrón
    let bluePieceCount = 0;
    for (let col = 0; col < BOARD_COLS; col++) {
        if (gameState.blueDistribution.pattern[col] === 'o') {
        gameState.board[blueStartRow][col].piece = {
            color: 'blue',
                id: `blue_${bluePieceCount}`
            };
            bluePieceCount++;
        }
    }
    
    console.log(`✅ Fichas colocadas: ${redPieceCount} rojas, ${bluePieceCount} azules`);
}

// Función para limpiar información de formaciones anteriores
function clearFormationInfo() {
    // Limpiar formaciones de la columna izquierda (CPU)
    const leftColumn = document.querySelector('.left-column');
    const existingCpuFormations = leftColumn.querySelectorAll('.formation-info.cpu-formation');
    existingCpuFormations.forEach(formation => formation.remove());
    
    // Limpiar formaciones de la columna derecha (Jugador)
    const rightColumn = document.querySelector('.right-column');
    const existingPlayerFormations = rightColumn.querySelectorAll('.formation-info.player-formation');
    existingPlayerFormations.forEach(formation => formation.remove());
    
    // Limpiar cualquier modal de formación abierto
    const existingModals = document.querySelectorAll('.formation-side-panel');
    existingModals.forEach(modal => modal.remove());
}

// Función para mostrar información de formaciones
function showFormationInfo() {
    if (!gameState.redDistribution || !gameState.blueDistribution) return;
    
    // Limpiar formaciones anteriores antes de crear las nuevas
    clearFormationInfo();
    
    // Crear columna de formación para CPU (izquierda)
    const leftColumn = document.querySelector('.left-column');
    const cpuFormationDiv = document.createElement('div');
    cpuFormationDiv.className = 'formation-info cpu-formation';
    cpuFormationDiv.innerHTML = `
        <div class="formation-details">
            <div class="formation-name">${gameState.redDistribution.name}</div>
            <div class="formation-visual-pattern">${createFormationVisual(gameState.redDistribution.pattern, 'red')}</div>
            <div class="formation-probability">${formatProbability(gameState.redDistribution.weight)}</div>
        </div>
    `;
    
    // Agregar evento click para mostrar modal
    cpuFormationDiv.addEventListener('click', () => {
        showFormationModal(gameState.redDistribution, 'CPU', 'red');
    });
    
    // Insertar al final de la columna izquierda
    leftColumn.appendChild(cpuFormationDiv);
    
    // Crear columna de formación para Jugador (derecha)
    const rightColumn = document.querySelector('.right-column');
    const playerFormationDiv = document.createElement('div');
    playerFormationDiv.className = 'formation-info player-formation';
    playerFormationDiv.innerHTML = `
        <div class="formation-details">
            <div class="formation-name">${gameState.blueDistribution.name}</div>
            <div class="formation-visual-pattern">${createFormationVisual(gameState.blueDistribution.pattern, 'blue')}</div>
            <div class="formation-probability">${formatProbability(gameState.blueDistribution.weight)}</div>
        </div>
    `;
    
    // Agregar evento click para mostrar modal
    playerFormationDiv.addEventListener('click', () => {
        showFormationModal(gameState.blueDistribution, 'Tu Formación', 'blue');
    });
    
    // Insertar al final de la columna derecha
    rightColumn.appendChild(playerFormationDiv);
}

// Función para crear el patrón visual con fichas
function createFormationVisual(pattern, teamColor) {
    const pieces = [];
    for (let i = 0; i < pattern.length; i++) {
        if (pattern[i] === 'o') {
            pieces.push(`<div class="formation-piece ${teamColor}-piece"></div>`);
        } else {
            pieces.push(`<div class="formation-empty"></div>`);
        }
    }
    return `<div class="formation-grid">${pieces.join('')}</div>`;
}

// Función para mostrar información detallada de formación en panel lateral
function showFormationModal(distribution, title, teamColor) {
    // Detectar el tema actual
    const isDarkMode = document.body.classList.contains('dark-theme');
    const themeClass = isDarkMode ? 'dark-theme' : 'light-theme';
    
    // Determinar el lado según el equipo (CPU=izquierda, Jugador=derecha)
    const sideClass = teamColor === 'red' ? 'formation-panel-left' : 'formation-panel-right';
    
    // No remover paneles existentes - permitir múltiples modales
    
    // Crear panel lateral
    const panel = document.createElement('div');
    panel.className = `formation-side-panel ${sideClass} ${themeClass}`;
    
    // Crear contenido del panel lateral
    panel.innerHTML = `
        <div class="formation-panel-header">
            <h3>${title}</h3>
            <button class="formation-panel-close">&times;</button>
        </div>
        
        <div class="formation-panel-body">
            <div class="formation-panel-name">${distribution.name}</div>
            <div class="formation-panel-pattern">${createFormationVisual(distribution.pattern, teamColor)}</div>
            <div class="formation-panel-probability">Probabilidad: ${formatProbability(distribution.weight)}</div>
            
            <div class="formation-panel-description">
                ${getFormationDescription(distribution.name)}
            </div>
            
            <div class="formation-panel-advantages">
                <h4>Ventajas:</h4>
                <ul>
                    ${getFormationAdvantages(distribution.name).map(advantage => `<li>${advantage}</li>`).join('')}
                </ul>
            </div>
            
            <div class="formation-panel-disadvantages">
                <h4>Desventajas:</h4>
                <ul>
                    ${getFormationDisadvantages(distribution.name).map(disadvantage => `<li>${disadvantage}</li>`).join('')}
                </ul>
            </div>
        </div>
    `;
    
    // Agregar al body
    document.body.appendChild(panel);
    
    // Animar entrada
    setTimeout(() => {
        panel.classList.add('show');
    }, 10);
    
    // Event listener para cerrar
    const closeBtn = panel.querySelector('.formation-panel-close');
    closeBtn.addEventListener('click', (event) => hideFormationModal(event));
}

// Función para ocultar información detallada de formación
function hideFormationModal(event) {
    // Encontrar el panel más cercano al botón que se clickeó
    const closeBtn = event.target;
    const panel = closeBtn.closest('.formation-side-panel');
    
    if (panel) {
        panel.classList.add('hide');
        setTimeout(() => {
            panel.remove();
        }, 300);
    }
}

// Función para obtener descripción de la formación
function getFormationDescription(name) {
    const descriptions = {
        "Aleatoria": "Distribución completamente aleatoria. Cada partida será única e impredecible, desafiando tanto al jugador como al rival.",

        // Formaciones del modo Express
        

        // Formaciones del modo Clásico
        "Centro Puro": "Distribución centrada que prioriza el control del centro del tablero. Estrategia ideal para jugadores defensivos que buscan estabilidad.",
        "Lateral Derecho": "Formación que concentra las piezas en el flanco derecho. Perfecta para jugadores agresivos que buscan sorprender al rival.",
        "Lateral Izquierdo": "Formación que concentra las piezas en el flanco izquierdo. Perfecta para jugadores agresivos que buscan sorprender al rival.",
        "Alternada Compacta": "Patrón alternado que mantiene las piezas compactas. Estrategia equilibrada que funciona bien en todas las situaciones.",
        "Centro con Flancos": "Estrategia que combina control central con presencia en los flancos. Para jugadores versátiles que se adaptan a cualquier estilo.",
        "Triple Centro": "Formación que refuerza el centro con tres piezas. Ideal para jugadores que buscan control total del juego.",
        
        // Formaciones del modo Master
        "Doble Núcleo": "Formación que divide el centro en dos grupos estratégicos. Permite control dual del centro del tablero.",
        "Fortaleza Lateral": "Concentración máxima en ambos laterales. Formación perfecta para empezar el juego con fuerza.",
        "Cuatro Carriles": "Cuatro núcleos de control uniformemente distribuidos. Estrategia de control dual muy efectiva.",
        "Cadena Alterna": "Patrón alternado que distribuye las piezas uniformemente de un modo conservador. Máxima flexibilidad táctica.",
        "Lateral Derecho": "Concentración total en el flanco derecho. Estrategia de ataque lateral agresiva.",
        "Lateral Izquierdo": "Concentración total en el flanco izquierdo. Estrategia de ataque lateral agresiva.",
        "Control Lateral": "Control estratégico de los flancos laterales con el centro protegido. Formación equilibrada entre ataque y defensa."
    };
    return descriptions[name] || "Formación especial con características únicas.";
}

// Función para obtener ventajas de la formación
function getFormationAdvantages(name) {
    const advantages = {
        "Centro Puro": [
            "Control central desde el inicio",
            "Defensa con apoyos abundantes"
        ],
        "Lateral Derecho": [
            "Ataque por el flanco derecho",
            "Acumulación de fuerzas en el flanco derecho"
        ],
        "Lateral Izquierdo": [
            "Ataque por el flanco izquierdo",
            "Acumulación de fuerzas en el flanco izquierdo"
        ],
        "Alternada Compacta": [
            "Distribución equilibrada",
            "Flexibilidad táctica"
        ],
        "Centro con Flancos": [
            "Presencia en múltiples zonas",
            "Opciones de ataque variadas"
        ],
        "Triple Centro": [
            "Fuerza concentrada en el centro",
            "Equilibrio entre el centro y los carriles"
        ],
        "Aleatoria": [
            "Impredecible para el rival",
            "Cada partida es única"
        ],
        // Ventajas del modo Master
        "Doble Núcleo": [
            "Flexibilidad de movimiento",
            "Balance equilibrado entre ataque y defensa"
        ],
        "Fortaleza Lateral": [
            "Dominio lateral absoluto",
            "Ataque con apoyos",
        ],
        "Cuatro Carriles": [
            "Dos puntos de control",
            "Mayor flexibilidad táctica",
        ],
        "Cadena Alterna": [
            "Cobertura uniforme",
            "Adaptable a cualquier situación"
        ],
        "Lateral Derecho": [
            "Ataque concentrado por la derecha",
            "Acumulación de fuerzas",
        ],
        "Lateral Izquierdo": [
            "Ataque concentrado por la izquierda",
            "Acumulación de fuerzas",
        ],
        "Control Lateral": [
            "Control de ambos flancos",
            "Múltiples opciones de ataque"
        ]
    };
    return advantages[name] || ["Características especiales", "Estrategia única"];
}

// Función para obtener desventajas de la formación
function getFormationDisadvantages(name) {
    const disadvantages = {
        "Centro Puro": [
            "Menos opciones de ataque lateral",
            "Formación predecible"
        ],
        "Lateral Derecho": [
            "Carril izquierdo menos protegido",
            "Movimientos necesarios para equilibrar la defensa"
        ],
        "Lateral Izquierdo": [
            "Carril derecho menos protegido",
            "Movimientos necesarios para equilibrar la defensa"
        ],
        "Alternada Compacta": [
            "Sin ventaja territorial específica",
            "Requiere adaptación constante"
        ],
        "Centro con Flancos": [
            "Fuerzas dispersas",
            "Requiere mayor coordinación"
        ],
        "Triple Centro": [
            "Menos flexibilidad táctica",
            "Vulnerable a ataques laterales"
        ],
        "Aleatoria": [
            "Sin control sobre la formación",
            "Puede resultar desequilibrada"
        ],
        // Desventajas del modo Master
        "Doble Núcleo": [
            "Fuerzas divididas",
            "Requiere coordinación precisa",
        ],
        "Fortaleza Lateral": [
            "Vulnerable a ataques centrales",
            "Ataques laterales previsibles",
        ],
        "Cuatro Carriles": [
            "Ataques previsibles",
            "Posibilidad de liberar carriles"
        ],
        "Cadena Alterna": [
            "Sin ventaja territorial específica",
            "Puede ser lenta en el desarrollo"
        ],
        "Lateral Derecho": [
            "Flanco izquierdo desprotegido",
            "Movimientos necesarios para equilibrar",
        ],
        "Lateral Izquierdo": [
            "Flanco derecho desprotegido",
            "Movimientos necesarios para equilibrar",
        ],
        "Control Lateral": [
            "Centro poco poblado",
            "Ataques laterales previsibles"
        ]
    };
    return disadvantages[name] || ["Desafíos únicos", "Limitaciones particulares"];
}

function createBoardHTML() {
    const boardElement = document.getElementById('gameBoard');
    boardElement.innerHTML = '';
    
    // Crear las filas del tablero
    for (let row = 0; row < BOARD_ROWS; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'board-row';
        
        // Para las filas de meta, crear una sola columna que ocupe todo el ancho
        if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
            const cell = gameState.board[row][0]; // Solo usamos la primera celda como referencia
            const cellElement = document.createElement('div');
            
            // Clases CSS para la celda de meta
            let className = `board-cell ${cell.type} goal-row`;
            
            // Agregar clase especial si hay una ficha seleccionada en posición de meta
            if (row === BLUE_GOAL_ROW && gameState.selectedPiece && gameState.selectedPiece.row === RED_START_ROW) {
                const selectedCell = gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col];
                if (selectedCell.piece && selectedCell.piece.color === 'blue') {
                    className += ' meta-available';
                }
            } else if (row === RED_GOAL_ROW && gameState.selectedPiece && gameState.selectedPiece.row === BLUE_START_ROW) {
                const selectedCell = gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col];
                if (selectedCell.piece && selectedCell.piece.color === 'red') {
                    className += ' meta-available';
                }
            }
            
            cellElement.className = className;
            cellElement.dataset.row = row;
            cellElement.dataset.col = 'all';
            
            // Agregar event listener para clicks en filas de meta
            cellElement.addEventListener('click', () => handleCellClick(row, 0));
            
            rowElement.appendChild(cellElement);
        } else {
            // Para el resto de filas, crear las 9 columnas normales
            for (let col = 0; col < BOARD_COLS; col++) {
                const cell = gameState.board[row][col];
                const cellElement = document.createElement('div');
                
                // Clases CSS para la celda
                cellElement.className = `board-cell ${cell.type}`;
                cellElement.dataset.row = row;
                cellElement.dataset.col = col;
                
                // Si hay una ficha en esta celda
                if (cell.piece) {
                    const pieceElement = document.createElement('div');
                    let pieceClass = `piece ${cell.piece.color}`;
                    
                    // Añadir clase de animación si la ficha se está moviendo
                    if (cell.piece.moving) {
                        pieceClass += ' moving';
                    }
                    
                    // Añadir clase de animación si la ficha se está eliminando
                    if (cell.piece.eliminating) {
                        pieceClass += ' eliminating';
                    }
                    
                    // Añadir clase de animación si la ficha está atacando
                    if (cell.piece.attacking) {
                        pieceClass += ' attacking';
                    }
                    
                    // Añadir clase de animación si la ficha no tiene movimientos
                    if (cell.piece.noMoves) {
                        pieceClass += ' no-moves';
                    }
                    
                    // Añadir clase de selección si la ficha está seleccionada
                    if (gameState.selectedPiece && gameState.selectedPiece.row === row && gameState.selectedPiece.col === col) {
                        pieceClass += ' selected';
                    }
                    
                    pieceElement.className = pieceClass;
                    pieceElement.dataset.pieceId = cell.piece.id;
                    cellElement.appendChild(pieceElement);
                }
                
                // Mostrar sugerencias de movimiento si están activas
                if (gameState.showingHints && gameState.hintMoves.some(move => move.row === row && move.col === col)) {
                    cellElement.classList.add('hint-cell');
                    // Crear elemento de punto para sugerencias
                    const hintDot = document.createElement('div');
                    hintDot.className = 'hint-dot';
                    cellElement.appendChild(hintDot);
                }
                
                // Agregar event listener para clicks
                cellElement.addEventListener('click', () => handleCellClick(row, col));
                
                rowElement.appendChild(cellElement);
            }
        }
        
        boardElement.appendChild(rowElement);
    }
}

function handleCellClick(row, col) {
    // Limpiar selección de ficha eliminada antes de cada turno del jugador
    clearInvalidSelection();
    
    // Si el juego ha terminado, ignoramos clicks
    if (gameState.gameEnded) {
        return;
    }

    // Si es turno de la CPU, ignoramos clicks
    if (gameState.currentPlayer === 'red') {
        console.log('Turno de la CPU. Espera a que termine.');
        return;
    }

    // Manejar clicks en filas de meta cuando hay una ficha seleccionada en posición
    if (row === BLUE_GOAL_ROW && gameState.selectedPiece && gameState.selectedPiece.row === RED_START_ROW) {
        const selectedCell = gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col];
        if (selectedCell.piece && selectedCell.piece.color === 'blue') {
            // Ficha llega a la meta - eliminar del tablero y aumentar contador
            gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col].piece = null;
            gameState.blueArrived += 1;
            gameState.bluePieces -= 1;
            gameState.bluePoints += 2; // 2 puntos por llegar a la meta
            gameState.selectedPiece = null;
            gameState.showingHints = false;
            gameState.hintMoves = [];
            audioManager.playGoalCelebration();
            
            // Actualizar interfaz
            createBoardHTML();
            updateGameInfo();
            
            // Fin de turno -> CPU
            gameState.currentPlayer = 'red';
            updateGameInfo();
            
            // La verificación de victoria se hace en el setTimeout para evitar problemas de timing
            
            // Pequeño delay para movimiento de CPU
            setTimeout(() => {
                if (!gameState.gameEnded) {
                    cpuMove();
                }
            }, 300);
            return;
        }
    }
    
    if (row === RED_GOAL_ROW && gameState.selectedPiece && gameState.selectedPiece.row === BLUE_START_ROW) {
        const selectedCell = gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col];
        if (selectedCell.piece && selectedCell.piece.color === 'red') {
            // Ficha llega a la meta - eliminar del tablero y aumentar contador
            gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col].piece = null;
            gameState.redArrived += 1;
            gameState.redPieces -= 1;
            gameState.redPoints += 2; // 2 puntos por llegar a la meta
            gameState.selectedPiece = null;
            gameState.showingHints = false;
            gameState.hintMoves = [];
            audioManager.playGoalCelebration();
            
            // Actualizar interfaz
            createBoardHTML();
            updateGameInfo();
            
            // Fin de turno -> CPU
            gameState.currentPlayer = 'red';
            updateGameInfo();
            
            // La verificación de victoria se hace en el setTimeout para evitar problemas de timing
            
            // Pequeño delay para movimiento de CPU
            setTimeout(() => {
                if (!gameState.gameEnded) {
                    cpuMove();
                }
            }, 300);
        return;
        }
    }

    const cell = gameState.board[row][col];

    // Si hay sugerencias mostrándose y se hace click en una casilla de sugerencia
    if (gameState.showingHints && gameState.hintMoves.some(move => move.row === row && move.col === col)) {
    const from = gameState.selectedPiece;
    const fromCell = gameState.board[from.row][from.col];

        // Verificar si está entrando a la meta azul
        if (row === BLUE_GOAL_ROW) {
            // Ficha llega a la meta - eliminar del tablero y aumentar contador
            gameState.board[from.row][from.col].piece = null;
            gameState.blueArrived += 1;
            gameState.bluePieces -= 1;
            gameState.bluePoints += 2; // Puntos por llegar a la meta
            gameState.selectedPiece = null;
            gameState.showingHints = false;
            gameState.hintMoves = [];
            audioManager.playGoalCelebration();
        } else {
            // Verificar si hay eliminación
            const toCell = gameState.board[row][col];
            if (toCell.piece && toCell.piece.color !== fromCell.piece.color) {
                // Limpiar sugerencias inmediatamente
                gameState.selectedPiece = null;
                gameState.showingHints = false;
                gameState.hintMoves = [];
                
                // Crear ficha eliminada con animación de muerte
                const eliminatingPiece = { ...toCell.piece, eliminating: true, eliminatingStartTime: Date.now() };
                gameState.board[row][col].piece = eliminatingPiece;
                
                // Actualizar interfaz para mostrar animación de muerte
                createBoardHTML();
                updateGameInfo();
                
                // Después de la animación de muerte, colocar ficha atacante
                setTimeout(() => {
                    if (fromCell.piece.color === 'blue') {
                        // Azul elimina roja
                        gameState.redPieces -= 1;
                        gameState.blueEliminated += 1;
                        gameState.bluePoints += 1;
                    } else {
                        // Roja elimina azul
                        gameState.bluePieces -= 1;
                        gameState.redEliminated += 1;
                        gameState.redPoints += 1;
                    }
                    
                    // Colocar ficha atacante en la casilla (sin animación)
                    const finalPiece = { ...fromCell.piece };
                    gameState.board[row][col].piece = finalPiece;
        gameState.board[from.row][from.col].piece = null;

                    // Limpiar cualquier selección que pueda estar en la ficha eliminada
                    if (gameState.selectedPiece && 
                        gameState.selectedPiece.row === row && 
                        gameState.selectedPiece.col === col) {
                        gameState.selectedPiece = null;
                        gameState.showingHints = false;
                        gameState.hintMoves = [];
                    }

        // Actualizar interfaz
        createBoardHTML();
        updateGameInfo();
                }, 400); // Duración de la animación de muerte
                
                // Determinar quién está eliminando para el sonido apropiado
                const isPlayerEliminating = fromCell.piece.color === 'blue';
                audioManager.playElimination(isPlayerEliminating);
            } else {
                // Limpiar sugerencias inmediatamente
        gameState.selectedPiece = null;
                gameState.showingHints = false;
                gameState.hintMoves = [];
                
                // Movimiento normal con animación
                const movingPiece = { ...fromCell.piece, moving: true };
                gameState.board[row][col].piece = movingPiece;
                gameState.board[from.row][from.col].piece = null;
        audioManager.playPieceMove();

                // Actualizar interfaz inmediatamente para mostrar la animación
        createBoardHTML();
        updateGameInfo();
                
                // Quitar la animación después de que termine
                setTimeout(() => {
                    if (gameState.board[row][col].piece) {
                        gameState.board[row][col].piece.moving = false;
                        createBoardHTML();
                    }
                }, 500);
            }
        }

        // Fin de turno -> CPU
        gameState.currentPlayer = 'red';
        updateGameInfo();

        // La verificación de victoria se hace en el setTimeout para evitar problemas de timing

        // Pequeño delay para movimiento de CPU
        setTimeout(() => {
            if (!gameState.gameEnded) {
                cpuMove();
            }
        }, 300);
        return;
    }

    // Selección inicial de ficha
    if (!gameState.selectedPiece) {
        if (cell.piece && cell.piece.color === 'blue') {
            // Verificar si la ficha tiene movimientos válidos
            const possibleMoves = getPossibleMoves({ row, col }, 'blue');
            const validMoves = possibleMoves.filter(move => isValidMove({ row, col }, move, 'blue'));
            
            if (validMoves.length > 0) {
                // La ficha tiene movimientos válidos
                gameState.selectedPiece = { row, col };
                showMoveHints({ row, col }, 'blue');
                playSound('select');
    } else {
                // La ficha no tiene movimientos válidos - mostrar animación
                const noMovesPiece = { ...cell.piece, noMoves: true };
                gameState.board[row][col].piece = noMovesPiece;
                createBoardHTML();
                playSound('error');
                
                // Quitar la animación después de que termine
                setTimeout(() => {
                    if (gameState.board[row][col].piece) {
                        gameState.board[row][col].piece.noMoves = false;
                        createBoardHTML();
                    }
                }, 600);
            }
        }
        return;
    }

    const from = gameState.selectedPiece;
    const fromCell = gameState.board[from.row][from.col];

    // Si clicas otra ficha tuya, cambias la selección
    if (cell.piece && cell.piece.color === 'blue') {
        // Limpiar selección anterior si existe
        if (gameState.selectedPiece) {
        gameState.selectedPiece = null;
            gameState.showingHints = false;
            gameState.hintMoves = [];
        }
        
        // Verificar si la ficha tiene movimientos válidos
        const possibleMoves = getPossibleMoves({ row, col }, 'blue');
        const validMoves = possibleMoves.filter(move => isValidMove({ row, col }, move, 'blue'));
        
        if (validMoves.length > 0) {
            // La ficha tiene movimientos válidos
            gameState.selectedPiece = { row, col };
            showMoveHints({ row, col }, 'blue');
            playSound('select');
        } else {
            // La ficha no tiene movimientos válidos - mostrar animación
            const noMovesPiece = { ...cell.piece, noMoves: true };
            gameState.board[row][col].piece = noMovesPiece;
            createBoardHTML();
        playSound('error');
            
            // Quitar la animación después de que termine
            setTimeout(() => {
                if (gameState.board[row][col].piece) {
                    gameState.board[row][col].piece.noMoves = false;
                    createBoardHTML();
                }
            }, 600);
        }
        return;
    }

    // Si se hace click en una casilla vacía que no es sugerencia, limpiar selección
    if (!cell.piece) {
        gameState.selectedPiece = null;
        gameState.showingHints = false;
        gameState.hintMoves = [];
        createBoardHTML();
        updateGameInfo();
        return;
    }
}

// Validación de movimientos según las reglas del juego
function isValidMove(from, to, color) {
    const dRow = to.row - from.row;
    const dCol = to.col - from.col;
    const fromCell = gameState.board[from.row][from.col];
    const toCell = gameState.board[to.row][to.col];

    // Verificar límites del tablero
    if (to.row < 0 || to.row >= BOARD_ROWS) return false;
    if (to.col < 0 || to.col >= BOARD_COLS) return false;

    // Verificar que no se salte fichas en el camino
    if (!isPathClear(from, to)) return false;

    // Verificar que la casilla de destino esté libre (excepto para metas y eliminaciones)
    if (toCell.piece && to.row !== 0 && to.row !== 10) {
        // No permitir moverse a una casilla donde hay una ficha que está siendo eliminada
        if (toCell.piece.eliminating) return false;
        
        // Solo permitir eliminación en campo propio
        if (!canEliminate(from, to, color)) return false;
    }

    // Reglas específicas por zona del tablero
    if (color === 'blue') {
        // Campo propio (filas 7-10, índices 6-9) - últimas 4 filas incluyendo aparición
        if (from.row >= 6 && from.row <= 9) {
            // En la fila de aparición (fila 10, índice 9): 1 o 2 casillas adelante o 1 o 2 hacia el lado
            if (from.row === BLUE_START_ROW) {
                const oneForward = dRow === -1 && dCol === 0;
                const twoForward = dRow === -2 && dCol === 0;
                const oneSide = dRow === 0 && Math.abs(dCol) === 1;
                const twoSide = dRow === 0 && Math.abs(dCol) === 2;
                return oneForward || twoForward || oneSide || twoSide;
            }
            // En las otras filas del campo propio: 1 adelante o 1 hacia el lado
            else {
                const oneForward = dRow === -1 && dCol === 0;
                const oneSide = dRow === 0 && Math.abs(dCol) === 1;
                return oneForward || oneSide;
            }
        }
        // Zona segura (fila 6, índice 5): 1 adelante o diagonal adelante (3 movimientos posibles)
        else if (from.row === 5) {
            const forward = dRow === -1 && dCol === 0;
            const diagonalLeft = dRow === -1 && dCol === -1;
            const diagonalRight = dRow === -1 && dCol === 1;
            return forward || diagonalLeft || diagonalRight;
        }
        // Campo contrario (filas 1-5, índices 0-4): 1 adelante o diagonal adelante
        else if (from.row >= 0 && from.row <= 4) {
            const forward = dRow === -1 && dCol === 0;
            const diagonalLeft = dRow === -1 && dCol === -1;
            const diagonalRight = dRow === -1 && dCol === 1;
            return forward || diagonalLeft || diagonalRight;
        }
        // Meta azul (fila 1, índice 0): no se puede mover desde aquí
        else if (from.row === BLUE_GOAL_ROW) {
            return false;
        }
        // Permitir movimiento a la meta azul desde fila 1 o desde fila de aparición roja (fila 1)
        else if (to.row === BLUE_GOAL_ROW) {
            // Se puede llegar a la meta desde la fila 1 (índice 1) con movimiento hacia adelante
            // O desde cualquier posición en la fila 1 (fila de aparición roja)
            return from.row === RED_START_ROW;
        }
    } else { // color === 'red'
        // Campo propio (filas 2-5, índices 1-4) - primeras 4 filas incluyendo aparición
        if (from.row >= 1 && from.row <= 4) {
            // En la fila de aparición (fila 2, índice 1): 1 o 2 casillas adelante o 1 o 2 hacia el lado
            if (from.row === RED_START_ROW) {
                const oneForward = dRow === 1 && dCol === 0;
                const twoForward = dRow === 2 && dCol === 0;
                const oneSide = dRow === 0 && Math.abs(dCol) === 1;
                const twoSide = dRow === 0 && Math.abs(dCol) === 2;
                return oneForward || twoForward || oneSide || twoSide;
            }
            // En las otras filas del campo propio: 1 adelante o 1 hacia el lado
            else {
                const oneForward = dRow === 1 && dCol === 0;
                const oneSide = dRow === 0 && Math.abs(dCol) === 1;
                return oneForward || oneSide;
            }
        }
        // Zona segura (fila 6, índice 5): 1 adelante o diagonal adelante (3 movimientos posibles)
        else if (from.row === 5) {
            const forward = dRow === 1 && dCol === 0;
            const diagonalLeft = dRow === 1 && dCol === -1;
            const diagonalRight = dRow === 1 && dCol === 1;
            return forward || diagonalLeft || diagonalRight;
        }
        // Campo contrario (filas 6-10, índices 5-9): 1 adelante o diagonal adelante
        else if (from.row >= 5 && from.row <= 9) {
            const forward = dRow === 1 && dCol === 0;
            const diagonalLeft = dRow === 1 && dCol === -1;
            const diagonalRight = dRow === 1 && dCol === 1;
            return forward || diagonalLeft || diagonalRight;
        }
        // Meta roja (fila 11, índice 10): no se puede mover desde aquí
        else if (from.row === RED_GOAL_ROW) {
            return false;
        }
        // Permitir movimiento a la meta roja desde fila 9 o desde fila de aparición azul (fila 9)
        else if (to.row === RED_GOAL_ROW) {
            // Se puede llegar a la meta desde la fila 9 (índice 9) con movimiento hacia adelante
            // O desde cualquier posición en la fila 9 (fila de aparición azul)
            return from.row === BLUE_START_ROW;
        }
    }

    return false;
}

// Función para verificar si el camino está libre (no hay fichas en el trayecto)
function isPathClear(from, to) {
    const dRow = to.row - from.row;
    const dCol = to.col - from.col;
    
    // Si es movimiento diagonal, verificar casillas intermedias
    if (Math.abs(dRow) === Math.abs(dCol) && Math.abs(dRow) > 1) {
        const steps = Math.abs(dRow);
        const stepRow = dRow / steps;
        const stepCol = dCol / steps;
        
        for (let i = 1; i < steps; i++) {
            const checkRow = from.row + (stepRow * i);
            const checkCol = from.col + (stepCol * i);
            if (gameState.board[checkRow][checkCol].piece) {
                return false;
            }
        }
    }
    
    // Si es movimiento recto (horizontal o vertical), verificar casillas intermedias
    if ((dRow === 0 && Math.abs(dCol) > 1) || (dCol === 0 && Math.abs(dRow) > 1)) {
        const steps = Math.max(Math.abs(dRow), Math.abs(dCol));
        const stepRow = dRow === 0 ? 0 : dRow / Math.abs(dRow);
        const stepCol = dCol === 0 ? 0 : dCol / Math.abs(dCol);
        
        for (let i = 1; i < steps; i++) {
            const checkRow = from.row + (stepRow * i);
            const checkCol = from.col + (stepCol * i);
            if (gameState.board[checkRow][checkCol].piece) {
                return false;
            }
        }
    }
    
    return true;
}

// Función para verificar si se puede eliminar una ficha
function canEliminate(from, to, color) {
    const toCell = gameState.board[to.row][to.col];
    
    // Solo se puede eliminar fichas del equipo contrario
    if (toCell.piece.color === color) return false;
    
    // No se puede eliminar en la zona segura (fila 5, índice 5)
    if (to.row === 5) return false;
    
    // Solo se puede eliminar en campo propio
    if (color === 'blue') {
        // Campo propio azul: filas 7-10 (índices 6-9)
        return to.row >= 6 && to.row <= 9;
    } else {
        // Campo propio rojo: filas 2-5 (índices 1-4)
        return to.row >= 1 && to.row <= 4;
    }
}

// Función para verificar si hay fichas en posición de meta
function hasPiecesInMetaPosition(color) {
    if (color === 'blue') {
        // Verificar si hay fichas azules en la fila 1 (fila de aparición roja)
        for (let col = 0; col < BOARD_COLS; col++) {
            const cell = gameState.board[1][col];
            if (cell.piece && cell.piece.color === 'blue') {
                return true;
            }
        }
    } else {
        // Verificar si hay fichas rojas en la fila 9 (fila de aparición azul)
        for (let col = 0; col < BOARD_COLS; col++) {
            const cell = gameState.board[9][col];
            if (cell.piece && cell.piece.color === 'red') {
                return true;
            }
        }
    }
    return false;
}

// Función para mostrar sugerencias de movimiento
function showMoveHints(from, color) {
    // Limpiar sugerencias anteriores
    gameState.showingHints = false;
    gameState.hintMoves = [];
    
    // Obtener movimientos posibles
    const possibleMoves = getPossibleMoves(from, color);
    const validMoves = possibleMoves.filter(move => isValidMove(from, move, color));
    
    // Guardar sugerencias
    gameState.hintMoves = validMoves;
    gameState.showingHints = true;
    
    // Actualizar interfaz para mostrar sugerencias
    createBoardHTML();
}

// Función para obtener todos los movimientos posibles de una ficha
function getPossibleMoves(from, color) {
    const moves = [];
    const { row, col } = from;

    if (color === 'blue') {
        // Campo propio (filas 7-10, índices 6-9)
        if (row >= 6 && row <= 9) {
            if (row === BLUE_START_ROW) {
                // Fila de aparición: 1 o 2 casillas adelante o 1 o 2 hacia el lado
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 2, col: col });
                moves.push({ row: row, col: col - 1 });
                moves.push({ row: row, col: col + 1 });
                moves.push({ row: row, col: col - 2 });
                moves.push({ row: row, col: col + 2 });
            } else {
                // Otras filas del campo propio: 1 adelante o 1 hacia el lado
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row, col: col - 1 });
                moves.push({ row: row, col: col + 1 });
            }
        }
        // Zona segura (fila 6, índice 5): 1 adelante o diagonal adelante
        else if (row === 5) {
            moves.push({ row: row - 1, col: col });
            moves.push({ row: row - 1, col: col - 1 });
            moves.push({ row: row - 1, col: col + 1 });
        }
        // Campo contrario (filas 1-5, índices 0-4): 1 adelante o diagonal adelante
        else if (row >= 0 && row <= 4) {
            moves.push({ row: row - 1, col: col });
            moves.push({ row: row - 1, col: col - 1 });
            moves.push({ row: row - 1, col: col + 1 });
        }
        // Meta azul (fila 1, índice 0): no se puede mover desde aquí
        else if (row === BLUE_GOAL_ROW) {
            // No se puede mover desde la meta
        }
        // Se puede llegar a la meta azul desde la fila 1 (índice 1) - fila de aparición roja
        if (row === RED_START_ROW) {
            moves.push({ row: 0, col: 0 }); // Meta azul
        }
    } else { // color === 'red'
        // Campo propio (filas 2-5, índices 1-4)
        if (row >= 1 && row <= 4) {
            if (row === RED_START_ROW) {
                // Fila de aparición: 1 o 2 casillas adelante o 1 o 2 hacia el lado
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 2, col: col });
                moves.push({ row: row, col: col - 1 });
                moves.push({ row: row, col: col + 1 });
                moves.push({ row: row, col: col - 2 });
                moves.push({ row: row, col: col + 2 });
            } else {
                // Otras filas del campo propio: 1 adelante o 1 hacia el lado
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row, col: col - 1 });
                moves.push({ row: row, col: col + 1 });
            }
        }
        // Zona segura (fila 6, índice 5): 1 adelante o diagonal adelante
        else if (row === 5) {
            moves.push({ row: row + 1, col: col });
            moves.push({ row: row + 1, col: col - 1 });
            moves.push({ row: row + 1, col: col + 1 });
        }
        // Campo contrario (filas 6-10, índices 5-9): 1 adelante o diagonal adelante
        else if (row >= 5 && row <= 9) {
            moves.push({ row: row + 1, col: col });
            moves.push({ row: row + 1, col: col - 1 });
            moves.push({ row: row + 1, col: col + 1 });
        }
        // Meta roja (fila 11, índice 10): no se puede mover desde aquí
        else if (row === RED_GOAL_ROW) {
            // No se puede mover desde la meta
        }
        // Se puede llegar a la meta roja desde la fila 9 (índice 9) - fila de aparición azul
        if (row === BLUE_START_ROW) {
            moves.push({ row: 10, col: 0 }); // Meta roja
        }
    }

    // Filtrar movimientos que estén fuera del tablero
    return moves.filter(move => 
        move.row >= 0 && move.row < BOARD_ROWS && 
        move.col >= 0 && move.col < BOARD_COLS
    );
}

// Sistema de IA con diferentes niveles de dificultad
const cpuAI = {
    // IA Principiante: Movimientos básicos con algunas decisiones inteligentes
    beginner: function(availablePieces) {
        // Prioridad 1: Buscar eliminaciones (pero no siempre las toma)
        const eliminationMoves = this.findEliminationMoves(availablePieces);
        if (eliminationMoves.length > 0 && Math.random() < 0.4) { // 40% de probabilidad de eliminar
            return eliminationMoves[Math.floor(Math.random() * eliminationMoves.length)];
        }
        
        // Prioridad 2: Avanzar hacia la meta de forma segura
        const safeAdvanceMoves = this.findSafeAdvanceMoves(availablePieces);
        if (safeAdvanceMoves.length > 0 && Math.random() < 0.6) { // 60% de probabilidad de avanzar de forma segura
            return safeAdvanceMoves[Math.floor(Math.random() * safeAdvanceMoves.length)];
        }
        
        // Prioridad 3: Avanzar hacia la meta (puede ser arriesgado)
        const advanceMoves = this.findAdvanceMoves(availablePieces);
        if (advanceMoves.length > 0 && Math.random() < 0.3) { // 30% de probabilidad de avanzar arriesgadamente
            return advanceMoves[Math.floor(Math.random() * advanceMoves.length)];
        }
        
        // Fallback: Movimiento aleatorio
        return this.getRandomMove(availablePieces);
    },
    
    // IA Intermedia: Siempre elimina, avanza estratégicamente con predicción básica
    intermediate: function(availablePieces) {
        // Prioridad 1: Buscar eliminaciones (siempre las toma)
        const eliminationMoves = this.findEliminationMoves(availablePieces);
        if (eliminationMoves.length > 0) {
            // Analizar cuál eliminación es más beneficiosa
            return this.analyzeEliminationMoves(eliminationMoves);
        }
        
        // Prioridad 2: Movimientos seguros hacia la meta con predicción
        const safeMoves = this.findSafeAdvanceMoves(availablePieces);
        if (safeMoves.length > 0) {
            return this.analyzeSafeMoves(safeMoves);
        }
        
        // Prioridad 3: Movimientos hacia la meta con análisis de riesgo
        const advanceMoves = this.findAdvanceMoves(availablePieces);
        if (advanceMoves.length > 0) {
            return this.analyzeAdvanceMoves(advanceMoves);
        }
        
        // Fallback: Movimiento aleatorio
        return this.getRandomMove(availablePieces);
    },
    
    // IA Experta: Defensa de meta prioritaria, eliminaciones y avance
    expert: function(availablePieces) {
        console.log('🧠 IA Experta analizando...');
        console.log('📊 Fichas disponibles:', availablePieces.length);
        
        // Prioridad 1: Eliminaciones estratégicas con análisis profundo
        const eliminationMoves = this.findEliminationMoves(availablePieces);
        console.log('🎯 Eliminaciones encontradas:', eliminationMoves.length);
        if (eliminationMoves.length > 0) {
            const boardAnalysis = this.analyzeBoardState();
            const move = this.analyzeExpertEliminations(eliminationMoves, boardAnalysis);
            console.log('⚔️ Eligiendo eliminación:', move);
            return move;
        }
        
        // Prioridad 2: Movimientos seguros hacia la meta
        const safeMoves = this.findSafeAdvanceMoves(availablePieces);
        console.log('🛡️ Movimientos seguros:', safeMoves.length);
        if (safeMoves.length > 0) {
            const move = this.analyzeSafeMoves(safeMoves);
            console.log('🛡️ Eligiendo movimiento seguro:', move);
            return move;
        }
        
        // Prioridad 3: Movimientos hacia la meta con análisis de riesgo
        const advanceMoves = this.findAdvanceMoves(availablePieces);
        console.log('📈 Movimientos de avance:', advanceMoves.length);
        if (advanceMoves.length > 0) {
            const move = this.analyzeAdvanceMoves(advanceMoves);
            console.log('📈 Eligiendo avance:', move);
            return move;
        }
        
        // Fallback: Movimiento aleatorio
        console.log('🎲 Usando movimiento aleatorio');
        return this.getRandomMove(availablePieces);
    },
    
    // Funciones auxiliares para encontrar tipos de movimientos
    findEliminationMoves: function(availablePieces) {
        const eliminationMoves = [];
        
        availablePieces.forEach(pieceData => {
            pieceData.moves.forEach(move => {
                const targetCell = gameState.board[move.row][move.col];
                if (targetCell.piece && targetCell.piece.color === 'blue') {
                    eliminationMoves.push({
                        from: pieceData.from,
                        to: move,
                        piece: pieceData.piece,
                        type: 'elimination'
                    });
                }
            });
        });
        
        return eliminationMoves;
    },
    
    findAdvanceMoves: function(availablePieces) {
        const advanceMoves = [];
        
        availablePieces.forEach(pieceData => {
            pieceData.moves.forEach(move => {
                // Para rojas, avanzar significa ir hacia abajo (aumentar row)
                if (move.row > pieceData.from.row) {
                    advanceMoves.push({
                        from: pieceData.from,
                        to: move,
                        piece: pieceData.piece,
                        type: 'advance'
                    });
                }
            });
        });
        
        return advanceMoves;
    },
    
    findSafeAdvanceMoves: function(availablePieces) {
        const safeAdvanceMoves = [];
        
        availablePieces.forEach(pieceData => {
            pieceData.moves.forEach(move => {
                // Avanzar y verificar que no esté en peligro inmediato
                if (move.row > pieceData.from.row) {
                    const targetCell = gameState.board[move.row][move.col];
                    
                    // Verificar que la casilla esté libre
                    if (!targetCell.piece) {
                        // Verificar que no haya fichas azules cerca que puedan eliminarla
                        const isSafe = !this.isPositionInDanger(move);
                        
                        // Verificar que no esté muy cerca de la zona de inicio azul (más seguro)
                        const distanceFromBlueStart = move.row;
                        
                        if (isSafe && distanceFromBlueStart > 2) {
                            safeAdvanceMoves.push({
                                from: pieceData.from,
                                to: move,
                                piece: pieceData.piece,
                                type: 'safe_advance',
                                safetyScore: distanceFromBlueStart // Mayor distancia = más seguro
                            });
                        }
                    }
                }
            });
        });
        
        // Ordenar por seguridad (más seguro primero)
        return safeAdvanceMoves.sort((a, b) => (b.safetyScore || 0) - (a.safetyScore || 0));
    },
    
    findDefensiveMoves: function(availablePieces) {
        // Por ahora, movimientos defensivos básicos
        // TODO: Implementar lógica defensiva más avanzada
        return [];
    },
    
    findStrategicAdvanceMoves: function(availablePieces) {
        // Por ahora, igual que advance moves
        // TODO: Implementar lógica estratégica más avanzada
        return this.findAdvanceMoves(availablePieces);
    },
    
    isPositionInDanger: function(position) {
        // Verificar si una posición está en peligro de ser eliminada por fichas azules
        // Las fichas azules pueden eliminar desde cualquier casilla adyacente
        
        // Verificar casillas adyacentes donde puede haber fichas azules
        const directions = [
            { row: -1, col: -1 }, { row: -1, col: 0 }, { row: -1, col: 1 },
            { row: 0, col: -1 },                          { row: 0, col: 1 },
            { row: 1, col: -1 },  { row: 1, col: 0 },  { row: 1, col: 1 }
        ];
        
        for (let dir of directions) {
            const checkRow = position.row + dir.row;
            const checkCol = position.col + dir.col;
            
            if (checkRow >= 0 && checkRow < BOARD_ROWS && checkCol >= 0 && checkCol < BOARD_COLS) {
                const checkCell = gameState.board[checkRow][checkCol];
                if (checkCell.piece && checkCell.piece.color === 'blue') {
                    // Verificar si la ficha azul puede moverse a nuestra posición
                    const bluePossibleMoves = getPossibleMoves({ row: checkRow, col: checkCol }, 'blue');
                    const canEliminate = bluePossibleMoves.some(move => 
                        move.row === position.row && move.col === position.col
                    );
                    
                    if (canEliminate) {
                        return true;
                    }
                }
            }
        }
        
        return false;
    },
    
    getRandomMove: function(availablePieces) {
        const randomPiece = availablePieces[Math.floor(Math.random() * availablePieces.length)];
        const randomMove = randomPiece.moves[Math.floor(Math.random() * randomPiece.moves.length)];
        
        return {
            from: randomPiece.from,
            to: randomMove,
            piece: randomPiece.piece,
            type: 'random'
        };
    },
    
    // Funciones de análisis para IA intermedia
    analyzeEliminationMoves: function(eliminationMoves) {
        // Evaluar cada eliminación y elegir la mejor
        const scoredMoves = eliminationMoves.map(move => {
            let score = 1; // Puntuación base por eliminar
            
            // Bonus por eliminar fichas avanzadas del rival
            if (move.to.row < 4) { // Fichas azules cerca de la meta roja
                score += 2;
            }
            
            // Bonus por eliminar desde una posición segura
            if (!this.isPositionInDanger(move.from)) {
                score += 1;
            }
            
            // Penalización si quedamos en peligro después de eliminar
            if (this.isPositionInDanger(move.to)) {
                score -= 1;
            }
            
            return { move, score };
        });
        
        // Ordenar por puntuación y elegir el mejor (con algo de aleatoriedad)
        scoredMoves.sort((a, b) => b.score - a.score);
        
        // 70% probabilidad de elegir el mejor, 30% de elegir entre los mejores
        const topMoves = scoredMoves.filter(m => m.score === scoredMoves[0].score);
        if (topMoves.length > 1 && Math.random() < 0.3) {
            return topMoves[Math.floor(Math.random() * topMoves.length)].move;
        }
        
        return scoredMoves[0].move;
    },
    
    analyzeSafeMoves: function(safeMoves) {
        // Evaluar movimientos seguros hacia la meta
        const scoredMoves = safeMoves.map(move => {
            let score = move.safetyScore || 0;
            
            // Bonus por avanzar más hacia la meta
            const progress = move.to.row - move.from.row;
            score += progress * 2;
            
            // Bonus por acercarse a la meta roja
            if (move.to.row > 7) {
                score += 3;
            }
            
            // Bonus por no estar en el centro (menos predecible)
            const centerDistance = Math.abs(move.to.col - 4);
            score += centerDistance * 0.5;
            
            return { move, score };
        });
        
        // Elegir el mejor movimiento seguro
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    analyzeAdvanceMoves: function(advanceMoves) {
        // Analizar movimientos hacia la meta con predicción de riesgo
        const scoredMoves = advanceMoves.map(move => {
            let score = 0;
            
            // Bonus por avanzar hacia la meta
            const progress = move.to.row - move.from.row;
            score += progress * 2;
            
            // Analizar qué pasará después de este movimiento
            const futureRisk = this.predictFutureRisk(move);
            score -= futureRisk * 3; // Penalización por riesgo futuro
            
            // Bonus por acercarse a la meta
            if (move.to.row > 6) {
                score += 2;
            }
            
            // Penalización si estamos en peligro inmediato
            if (this.isPositionInDanger(move.to)) {
                score -= 2;
            }
            
            return { move, score };
        });
        
        // Filtrar movimientos muy arriesgados (solo 60% de probabilidad de tomarlos)
        const safeMoves = scoredMoves.filter(m => m.score > -2);
        const riskyMoves = scoredMoves.filter(m => m.score <= -2);
        
        if (safeMoves.length > 0 && Math.random() < 0.6) {
            // Elegir entre movimientos seguros
            safeMoves.sort((a, b) => b.score - a.score);
            return safeMoves[0].move;
        } else if (riskyMoves.length > 0) {
            // A veces tomar movimientos arriesgados (para no ser demasiado perfecta)
            riskyMoves.sort((a, b) => b.score - a.score);
            return riskyMoves[0].move;
        } else if (scoredMoves.length > 0) {
            // Fallback a cualquier movimiento
            scoredMoves.sort((a, b) => b.score - a.score);
            return scoredMoves[0].move;
        }
        
        return advanceMoves[0];
    },
    
    predictFutureRisk: function(move) {
        // Predicción básica: analizar si el rival puede eliminarnos en su próximo turno
        let riskLevel = 0;
        
        // Verificar fichas azules cercanas que puedan atacarnos
        const directions = [
            { row: -1, col: -1 }, { row: -1, col: 0 }, { row: -1, col: 1 },
            { row: 0, col: -1 },                          { row: 0, col: 1 },
            { row: 1, col: -1 },  { row: 1, col: 0 },  { row: 1, col: 1 }
        ];
        
        for (let dir of directions) {
            const checkRow = move.to.row + dir.row;
            const checkCol = move.to.col + dir.col;
            
            if (checkRow >= 0 && checkRow < BOARD_ROWS && checkCol >= 0 && checkCol < BOARD_COLS) {
                const checkCell = gameState.board[checkRow][checkCol];
                if (checkCell.piece && checkCell.piece.color === 'blue') {
                    // Verificar si esta ficha azul puede moverse a nuestra posición
                    const bluePossibleMoves = getPossibleMoves({ row: checkRow, col: checkCol }, 'blue');
                    const canEliminate = bluePossibleMoves.some(blueMove => 
                        blueMove.row === move.to.row && blueMove.col === move.to.col
                    );
                    
                    if (canEliminate) {
                        riskLevel += 2; // Alto riesgo si nos pueden eliminar
                        
                        // Riesgo adicional si la ficha azul está en una buena posición
                        if (checkRow < 4) { // Fichas azules avanzadas
                            riskLevel += 1;
                        }
                    }
                }
            }
        }
        
        return riskLevel;
    },
    
    // Funciones avanzadas para IA experta
    analyzeBoardState: function() {
        const analysis = {
            redPieces: [],
            bluePieces: [],
            redThreats: [],
            blueThreats: [],
            redAdvancement: 0,
            blueAdvancement: 0,
            boardControl: { red: 0, blue: 0 }
        };
        
        // Analizar todas las fichas y amenazas
        for (let row = 0; row < BOARD_ROWS; row++) {
            for (let col = 0; col < BOARD_COLS; col++) {
                const cell = gameState.board[row][col];
                if (cell.piece && !cell.piece.eliminating) {
                    const piece = cell.piece;
                    const position = { row, col };
                    
                    if (piece.color === 'red') {
                        analysis.redPieces.push(position);
                        analysis.redAdvancement += row;
                        
                        // Verificar amenazas hacia fichas azules
                        const threats = this.findThreatsFrom(position, 'red');
                        analysis.redThreats.push(...threats);
                    } else if (piece.color === 'blue') {
                        analysis.bluePieces.push(position);
                        analysis.blueAdvancement += (BOARD_ROWS - 1 - row);
                        
                        // Verificar amenazas hacia fichas rojas
                        const threats = this.findThreatsFrom(position, 'blue');
                        analysis.blueThreats.push(...threats);
                    }
                }
            }
        }
        
        analysis.boardControl.red = analysis.redPieces.length;
        analysis.boardControl.blue = analysis.bluePieces.length;
        
        return analysis;
    },
    
    findThreatsFrom: function(position, color) {
        const threats = [];
        const possibleMoves = getPossibleMoves(position, color);
        
        for (let move of possibleMoves) {
            const targetCell = gameState.board[move.row][move.col];
            if (targetCell.piece && targetCell.piece.color !== color && !targetCell.piece.eliminating) {
                threats.push({
                    from: position,
                    to: move,
                    target: targetCell.piece
                });
            }
        }
        
        return threats;
    },
    
    analyzeExpertEliminations: function(eliminationMoves, boardAnalysis) {
        const scoredMoves = eliminationMoves.map(move => {
            let score = 10; // Puntuación base alta por eliminar
            
            // DEFENSA DE META - Prioridad máxima
            // Para fichas azules, defender meta azul (fila 0)
            if (move.piece.color === 'blue') {
                const targetRow = move.to.row;
                if (targetRow <= 2) { // Fichas rojas cerca de meta azul
                    score += 20; // Valor muy alto por defensa de meta
                    score += (3 - targetRow) * 5; // Fila 0 = +15, Fila 1 = +10, Fila 2 = +5
                    
                    // Bonus si está en el centro
                    if (move.to.col >= 3 && move.to.col <= 5) {
                        score += 3;
                    }
                }
            }
            
            // Para fichas rojas, defender meta roja (fila 9)
            if (move.piece.color === 'red') {
                const targetRow = move.to.row;
                if (targetRow >= 7) { // Fichas azules cerca de meta roja
                    score += 20; // Valor muy alto por defensa de meta
                    score += (targetRow - 7) * 5; // Fila 9 = +10, Fila 8 = +5
                    
                    // Bonus si está en el centro
                    if (move.to.col >= 3 && move.to.col <= 5) {
                        score += 3;
                    }
                }
            }
            
            // Eliminaciones estratégicas generales
            const targetRow = move.to.row;
            if (move.piece.color === 'red' && targetRow < 3) {
                score += 5; // Fichas azules muy avanzadas
            } else if (move.piece.color === 'blue' && targetRow > 6) {
                score += 5; // Fichas rojas muy avanzadas
            }
            
            // Bonus por eliminar desde posición segura
            if (!this.isPositionInDanger(move.from)) {
                score += 3;
            }
            
            // Penalización si quedamos en peligro después
            if (this.isPositionInDanger(move.to)) {
                score -= 4;
            }
            
            // Bonus por eliminar fichas que están amenazando nuestras fichas
            const targetThreats = boardAnalysis.redThreats.filter(threat => 
                threat.to.row === move.to.row && threat.to.col === move.to.col
            );
            score += targetThreats.length * 2;
            
            return { move, score };
        });
        
        // Elegir el mejor movimiento de eliminación
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    findExpertDefensiveMoves: function(availablePieces, boardAnalysis) {
        const defensiveMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'defensive'
                };
                
                let defensiveValue = 0;
                
                // Bloquear avance del rival
                defensiveValue += this.evaluateBlockingValue(move, boardAnalysis);
                
                // Proteger nuestras fichas avanzadas
                defensiveValue += this.evaluateProtectionValue(move, boardAnalysis);
                
                if (defensiveValue > 0) {
                    moveObj.defensiveValue = defensiveValue;
                    defensiveMoves.push(moveObj);
                }
            }
        }
        
        return defensiveMoves.sort((a, b) => b.defensiveValue - a.defensiveValue);
    },
    
    evaluateBlockingValue: function(move, boardAnalysis) {
        let value = 0;
        
        // Bloquear fichas rojas que intentan avanzar hacia la meta azul
        if (move.piece.color === 'blue') {
            const redPiecesNearby = boardAnalysis.redPieces.filter(redPos => {
                const distance = Math.abs(redPos.row - move.to.row) + Math.abs(redPos.col - move.to.col);
                return distance <= 2 && redPos.row > move.to.row;
            });
            
            value += redPiecesNearby.length * 2;
            
            // Bonus por bloquear en el centro del tablero
            if (move.to.row >= 3 && move.to.row <= 6 && move.to.col >= 2 && move.to.col <= 6) {
                value += 1;
            }
        }
        
        return value;
    },
    
    evaluateProtectionValue: function(move, boardAnalysis) {
        let value = 0;
        
        // Proteger nuestras fichas avanzadas
        const ourAdvancedPieces = move.piece.color === 'red' 
            ? boardAnalysis.redPieces.filter(pos => pos.row > 5)
            : boardAnalysis.bluePieces.filter(pos => pos.row < 4);
        
        for (let advancedPiece of ourAdvancedPieces) {
            const distance = Math.abs(advancedPiece.row - move.to.row) + Math.abs(advancedPiece.col - move.to.col);
            if (distance === 1) {
                value += 3; // Protección directa
            } else if (distance === 2) {
                value += 1; // Protección indirecta
            }
        }
        
        return value;
    },
    
    analyzeDefensiveMoves: function(defensiveMoves, boardAnalysis) {
        const scoredMoves = defensiveMoves.map(move => {
            let score = move.defensiveValue || 0;
            
            // Bonus por mantener seguridad
            if (!this.isPositionInDanger(move.to)) {
                score += 2;
            }
            
            // Penalización por ponerse en peligro
            if (this.isPositionInDanger(move.to)) {
                score -= 3;
            }
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    findExpertSafeAdvances: function(availablePieces, boardAnalysis) {
        const safeAdvances = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'safe_advance'
                };
                
                // Verificar que el movimiento sea seguro
                const isSafe = this.isMoveSafe(move, boardAnalysis);
                if (isSafe) {
                    // Calcular valor de avance
                    const advanceValue = this.calculateAdvanceValue(move, piece);
                    moveObj.advanceValue = advanceValue;
                    safeAdvances.push(moveObj);
                }
            }
        }
        
        return safeAdvances.sort((a, b) => b.advanceValue - a.advanceValue);
    },
    
    isMoveSafe: function(move, boardAnalysis) {
        // Verificar que no estamos en peligro inmediato
        if (this.isPositionInDanger(move.to)) {
            return false;
        }
        
        // Verificar que no podemos ser eliminados en el siguiente turno
        const futureRisk = this.predictFutureRisk(move);
        if (futureRisk > 2) {
            return false;
        }
        
        // Verificar que no bloqueamos nuestras propias fichas
        const ourPieces = move.piece.color === 'red' ? boardAnalysis.redPieces : boardAnalysis.bluePieces;
        const blocksOurPieces = ourPieces.some(pos => 
            pos.row === move.to.row && pos.col === move.to.col
        );
        
        return !blocksOurPieces;
    },
    
    calculateAdvanceValue: function(move, piece) {
        let value = 0;
        
        // Valor base por avanzar
        if (piece.color === 'red') {
            value += (move.to.row - move.from.row) * 2; // Avanzar hacia abajo
        } else {
            value += (move.from.row - move.to.row) * 2; // Avanzar hacia arriba
        }
        
        // Bonus por acercarse a la meta
        if (piece.color === 'red' && move.to.row > 6) {
            value += 5;
        } else if (piece.color === 'blue' && move.to.row < 3) {
            value += 5;
        }
        
        // Bonus por posicionarse estratégicamente
        if (move.to.col >= 2 && move.to.col <= 6 && move.to.row >= 3 && move.to.row <= 6) {
            value += 1; // Centro del tablero
        }
        
        return value;
    },
    
    analyzeSafeAdvances: function(safeAdvances, boardAnalysis) {
        const scoredMoves = safeAdvances.map(move => {
            let score = move.advanceValue || 0;
            
            // Bonus por mantener ventaja posicional
            const positionalAdvantage = this.evaluatePositionalAdvantage(move, boardAnalysis);
            score += positionalAdvantage;
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    evaluatePositionalAdvantage: function(move, boardAnalysis) {
        let advantage = 0;
        
        // Ventaja por controlar el centro
        if (move.to.row >= 3 && move.to.row <= 6 && move.to.col >= 2 && move.to.col <= 6) {
            advantage += 1;
        }
        
        // Ventaja por tener fichas en posiciones avanzadas
        const ourPieces = move.piece.color === 'red' ? boardAnalysis.redPieces : boardAnalysis.bluePieces;
        const advancedPieces = ourPieces.filter(pos => {
            if (move.piece.color === 'red') {
                return pos.row > 5;
            } else {
                return pos.row < 4;
            }
        });
        
        advantage += advancedPieces.length * 0.5;
        
        return advantage;
    },
    
    analyzeStrategicMoves: function(strategicMoves, boardAnalysis) {
        const scoredMoves = strategicMoves.map(move => {
            let score = 0;
            
            // Evaluar valor estratégico
            score += this.evaluateStrategicValue(move, boardAnalysis);
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    evaluateStrategicValue: function(move, boardAnalysis) {
        let value = 0;
        
        // Valor por avanzar hacia la meta
        if (move.piece.color === 'red') {
            value += (move.to.row - move.from.row) * 1.5;
        } else {
            value += (move.from.row - move.to.row) * 1.5;
        }
        
        // Valor por posicionamiento
        if (move.to.col >= 2 && move.to.col <= 6) {
            value += 0.5;
        }
        
        return value;
    },
    
    // Funciones avanzadas para IA experta mejorada
    findTrapMoves: function(availablePieces, boardAnalysis) {
        const trapMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'trap'
                };
                
                let trapValue = 0;
                
                // Trampa 1: Sacrificar ficha para que otra llegue a meta
                trapValue += this.evaluateSacrificeForGoal(move, boardAnalysis);
                
                // Trampa 2: Dejar pasar para luego eliminar
                trapValue += this.evaluateLureAndEliminate(move, boardAnalysis);
                
                if (trapValue > 0) {
                    moveObj.trapValue = trapValue;
                    trapMoves.push(moveObj);
                }
            }
        }
        
        return trapMoves.sort((a, b) => b.trapValue - a.trapValue);
    },
    
    evaluateSacrificeForGoal: function(move, boardAnalysis) {
        let value = 0;
        
        // Si movemos una ficha que está bloqueando el camino de otra ficha hacia la meta
        if (move.piece.color === 'blue') {
            // Verificar si al movernos liberamos el camino para otra ficha azul
            const ourAdvancedPieces = boardAnalysis.bluePieces.filter(pos => pos.row > 5);
            
            for (let advancedPiece of ourAdvancedPieces) {
                // Verificar si nuestra ficha está bloqueando el camino
                const isBlocking = this.isBlockingPath(move.from, advancedPiece);
                if (isBlocking) {
                    // Verificar si al movernos liberamos el camino hacia la meta
                    const pathToGoal = this.getPathToGoal(advancedPiece, 'blue');
                    const isPathCleared = this.isPathClearedAfterMove(move, pathToGoal);
                    
                    if (isPathCleared) {
                        value += 8; // Alto valor por liberar camino hacia meta
                        
                        // Bonus si la ficha liberada está muy cerca de la meta
                        if (advancedPiece.row > 7) {
                            value += 5;
                        }
                    }
                }
            }
        }
        
        return value;
    },
    
    evaluateLureAndEliminate: function(move, boardAnalysis) {
        let value = 0;
        
        // Crear una trampa donde el rival puede eliminar nuestra ficha pero luego podemos eliminarlo
        if (this.isPositionInDanger(move.to)) {
            // Verificar si tenemos fichas que pueden eliminar al rival después de que nos elimine
            const enemyPieces = move.piece.color === 'blue' ? boardAnalysis.redPieces : boardAnalysis.bluePieces;
            
            for (let enemyPos of enemyPieces) {
                // Verificar si el enemigo puede eliminarnos desde esta posición
                const canEliminateUs = this.canEliminateFrom(enemyPos, move.to, move.piece.color === 'blue' ? 'red' : 'blue');
                
                if (canEliminateUs) {
                    // Verificar si tenemos fichas que pueden eliminar al enemigo después
                    const ourPieces = move.piece.color === 'blue' ? boardAnalysis.bluePieces : boardAnalysis.redPieces;
                    
                    for (let ourPos of ourPieces) {
                        if (ourPos.row !== move.from.row || ourPos.col !== move.from.col) {
                            const canEliminateEnemy = this.canEliminateFrom(ourPos, enemyPos, move.piece.color);
                            
                            if (canEliminateEnemy) {
                                value += 6; // Trampa de atracción y eliminación
                                
                                // Bonus si el enemigo está en una posición valiosa
                                if (enemyPos.row > 6 || enemyPos.row < 3) {
                                    value += 2;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return value;
    },
    
    analyzeTrapMoves: function(trapMoves, boardAnalysis) {
        const scoredMoves = trapMoves.map(move => {
            let score = move.trapValue || 0;
            
            // Bonus por trampas que no nos ponen en peligro inmediato
            if (!this.isPositionInDanger(move.to)) {
                score += 2;
            }
            
            // Penalización por trampas muy arriesgadas
            const futureRisk = this.predictFutureRisk(move);
            score -= futureRisk;
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    // Funciones auxiliares para trampas y estrategias avanzadas
    isBlockingPath: function(from, target) {
        // Verificar si una posición está bloqueando el camino hacia otra
        const rowDiff = target.row - from.row;
        const colDiff = target.col - from.col;
        
        // Si están en la misma fila o columna y la distancia es pequeña
        return (rowDiff === 0 && Math.abs(colDiff) <= 2) || 
               (colDiff === 0 && Math.abs(rowDiff) <= 2);
    },
    
    getPathToGoal: function(position, color) {
        // Calcular la ruta más directa hacia la meta
        const goalRow = color === 'blue' ? 0 : BOARD_ROWS - 1;
        const path = [];
        
        let currentRow = position.row;
        let currentCol = position.col;
        
        while (currentRow !== goalRow) {
            if (color === 'blue') {
                currentRow--;
            } else {
                currentRow++;
            }
            path.push({ row: currentRow, col: currentCol });
        }
        
        return path;
    },
    
    isPathClearedAfterMove: function(move, path) {
        // Verificar si al hacer un movimiento se libera un camino
        return !path.some(pathPos => 
            pathPos.row === move.from.row && pathPos.col === move.from.col
        );
    },
    
    canEliminateFrom: function(from, target, color) {
        // Verificar si una ficha puede eliminar a otra desde su posición
        const possibleMoves = getPossibleMoves(from, color);
        return possibleMoves.some(move => 
            move.row === target.row && move.col === target.col
        );
    },
    
    findAdvancedDefensiveMoves: function(availablePieces, boardAnalysis) {
        const defensiveMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'advanced_defensive'
                };
                
                let defensiveValue = 0;
                
                // Defensa avanzada: Dejar pasar para luego eliminar
                defensiveValue += this.evaluateAdvancedBlocking(move, boardAnalysis);
                
                if (defensiveValue > 0) {
                    moveObj.defensiveValue = defensiveValue;
                    defensiveMoves.push(moveObj);
                }
            }
        }
        
        return defensiveMoves.sort((a, b) => b.defensiveValue - a.defensiveValue);
    },
    
    evaluateAdvancedBlocking: function(move, boardAnalysis) {
        let value = 0;
        
        // Dejar pasar una ficha para luego eliminarla desde una mejor posición
        if (move.piece.color === 'blue') {
            const redPieces = boardAnalysis.redPieces;
            
            for (let redPos of redPieces) {
                // Verificar si al movernos permitimos que una ficha roja avance
                // pero creamos una mejor posición para eliminarla después
                const redPossibleMoves = getPossibleMoves(redPos, 'red');
                
                for (let redMove of redPossibleMoves) {
                    // Verificar si el movimiento rojo nos daría una mejor posición de ataque
                    const newAttackPosition = this.getBetterAttackPosition(move.to, redMove);
                    
                    if (newAttackPosition) {
                        value += 3; // Valor por crear mejor posición de ataque
                        
                        // Bonus si la ficha roja está avanzada
                        if (redPos.row > 5) {
                            value += 2;
                        }
                    }
                }
            }
        }
        
        return value;
    },
    
    getBetterAttackPosition: function(ourMove, enemyMove) {
        // Verificar si al movernos a una posición podemos atacar mejor al enemigo
        const distance = Math.abs(ourMove.row - enemyMove.row) + Math.abs(ourMove.col - enemyMove.col);
        return distance <= 2; // Si estamos a distancia de ataque
    },
    
    analyzeAdvancedDefensiveMoves: function(defensiveMoves, boardAnalysis) {
        const scoredMoves = defensiveMoves.map(move => {
            let score = move.defensiveValue || 0;
            
            // Bonus por defensas que no comprometen nuestras fichas
            if (!this.isPositionInDanger(move.to)) {
                score += 2;
            }
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    findFirstRowMaintenanceMoves: function(availablePieces, boardAnalysis) {
        const firstRowMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            // Solo considerar fichas que están en la primera fila o pueden volver a ella
            const isInFirstRow = (piece.color === 'blue' && from.row === BLUE_GOAL_ROW) || 
                                (piece.color === 'red' && from.row === BOARD_ROWS - 1);
            const canReturnToFirstRow = (piece.color === 'blue' && pieceData.moves.some(m => m.row === BLUE_GOAL_ROW)) ||
                                       (piece.color === 'red' && pieceData.moves.some(m => m.row === BOARD_ROWS - 1));
            
            if (isInFirstRow || canReturnToFirstRow) {
                for (let move of pieceData.moves) {
                    const moveObj = {
                        from: from,
                        to: move,
                        piece: piece,
                        type: 'first_row_maintenance'
                    };
                    
                    let firstRowValue = 0;
                    
                    // Valor por mantener fichas en primera fila
                    if ((piece.color === 'blue' && move.row === BLUE_GOAL_ROW) || 
                        (piece.color === 'red' && move.row === BOARD_ROWS - 1)) {
                        firstRowValue += 5; // Alto valor por primera fila
                        
                        // Bonus por tener múltiples fichas en primera fila
                        const firstRowPieces = piece.color === 'blue' ? 
                            boardAnalysis.bluePieces.filter(pos => pos.row === BLUE_GOAL_ROW).length :
                            boardAnalysis.redPieces.filter(pos => pos.row === BOARD_ROWS - 1).length;
                        
                        firstRowValue += firstRowPieces * 2;
                    }
                    
                    if (firstRowValue > 0) {
                        moveObj.firstRowValue = firstRowValue;
                        firstRowMoves.push(moveObj);
                    }
                }
            }
        }
        
        return firstRowMoves.sort((a, b) => b.firstRowValue - a.firstRowValue);
    },
    
    analyzeFirstRowMoves: function(firstRowMoves, boardAnalysis) {
        const scoredMoves = firstRowMoves.map(move => {
            let score = move.firstRowValue || 0;
            
            // Bonus por mantener seguridad
            if (!this.isPositionInDanger(move.to)) {
                score += 2;
            }
            
            // Bonus por controlar el centro desde primera fila
            if (move.to.col >= 3 && move.to.col <= 5) {
                score += 1;
            }
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    // Funciones de defensa de meta para IA experta
    findMetaDefenseMoves: function(availablePieces, boardAnalysis) {
        const metaDefenseMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'meta_defense'
                };
                
                let defenseValue = 0;
                
                // Verificar si podemos eliminar fichas que están cerca de nuestra meta
                defenseValue += this.evaluateMetaDefense(move, boardAnalysis);
                
                if (defenseValue > 0) {
                    moveObj.defenseValue = defenseValue;
                    metaDefenseMoves.push(moveObj);
                }
            }
        }
        
        return metaDefenseMoves.sort((a, b) => b.defenseValue - a.defenseValue);
    },
    
    evaluateMetaDefense: function(move, boardAnalysis) {
        let value = 0;
        
        // Para fichas azules, defender la meta azul (fila 0)
        if (move.piece.color === 'blue') {
            // Buscar fichas rojas que están cerca de la meta azul
            const redPiecesNearBlueGoal = boardAnalysis.redPieces.filter(pos => pos.row <= 2);
            
            for (let redPos of redPiecesNearBlueGoal) {
                // Verificar si podemos eliminar esta ficha roja
                const canEliminate = this.canEliminateFrom(move.to, redPos, 'blue');
                
                if (canEliminate) {
                    // Valor muy alto por eliminar fichas cerca de nuestra meta
                    let eliminationValue = 20; // Valor base muy alto
                    
                    // Bonus por distancia a la meta (más cerca = más peligroso)
                    eliminationValue += (3 - redPos.row) * 5; // Fila 0 = +15, Fila 1 = +10, Fila 2 = +5
                    
                    // Bonus si la ficha roja está en el centro (más fácil que llegue)
                    if (redPos.col >= 3 && redPos.col <= 5) {
                        eliminationValue += 3;
                    }
                    
                    value += eliminationValue;
                }
            }
        }
        
        // Para fichas rojas, defender la meta roja (fila 9)
        if (move.piece.color === 'red') {
            // Buscar fichas azules que están cerca de la meta roja
            const bluePiecesNearRedGoal = boardAnalysis.bluePieces.filter(pos => pos.row >= 7);
            
            for (let bluePos of bluePiecesNearRedGoal) {
                // Verificar si podemos eliminar esta ficha azul
                const canEliminate = this.canEliminateFrom(move.to, bluePos, 'red');
                
                if (canEliminate) {
                    // Valor muy alto por eliminar fichas cerca de nuestra meta
                    let eliminationValue = 20; // Valor base muy alto
                    
                    // Bonus por distancia a la meta (más cerca = más peligroso)
                    eliminationValue += (bluePos.row - 7) * 5; // Fila 9 = +10, Fila 8 = +5
                    
                    // Bonus si la ficha azul está en el centro (más fácil que llegue)
                    if (bluePos.col >= 3 && bluePos.col <= 5) {
                        eliminationValue += 3;
                    }
                    
                    value += eliminationValue;
                }
            }
        }
        
        return value;
    },
    
    analyzeMetaDefenseMoves: function(metaDefenseMoves, boardAnalysis) {
        const scoredMoves = metaDefenseMoves.map(move => {
            let score = move.defenseValue || 0;
            
            // Bonus por no ponernos en peligro después de eliminar
            if (!this.isPositionInDanger(move.to)) {
                score += 5;
            }
            
            // Penalización por ponernos en peligro
            if (this.isPositionInDanger(move.to)) {
                score -= 3;
            }
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    },
    
    findBlockingMoves: function(availablePieces, boardAnalysis) {
        const blockMoves = [];
        
        for (let pieceData of availablePieces) {
            const piece = pieceData.piece;
            const from = pieceData.from;
            
            for (let move of pieceData.moves) {
                const moveObj = {
                    from: from,
                    to: move,
                    piece: piece,
                    type: 'blocking'
                };
                
                let blockValue = 0;
                
                // Verificar si podemos bloquear fichas que están avanzando hacia nuestra meta
                blockValue += this.evaluateBlocking(move, boardAnalysis);
                
                if (blockValue > 0) {
                    moveObj.blockValue = blockValue;
                    blockMoves.push(moveObj);
                }
            }
        }
        
        return blockMoves.sort((a, b) => b.blockValue - a.blockValue);
    },
    
    evaluateBlocking: function(move, boardAnalysis) {
        let value = 0;
        
        // Para fichas azules, bloquear fichas rojas que avanzan hacia la meta azul
        if (move.piece.color === 'blue') {
            const redPieces = boardAnalysis.redPieces;
            
            for (let redPos of redPieces) {
                // Verificar si la ficha roja está avanzando hacia la meta azul
                if (redPos.row <= 4) { // Fichas rojas en la mitad superior
                    // Calcular si nuestro movimiento bloquea una ruta de avance
                    const blocksRoute = this.blocksAdvanceRoute(move.to, redPos, 'red');
                    
                    if (blocksRoute) {
                        // Valor por bloquear, mayor si la ficha está más cerca de la meta
                        let blockValue = 10; // Valor base
                        
                        // Bonus por distancia a la meta
                        blockValue += (5 - redPos.row) * 2; // Fila 0 = +10, Fila 1 = +8, etc.
                        
                        // Bonus si bloqueamos en el centro
                        if (move.to.col >= 3 && move.to.col <= 5) {
                            blockValue += 2;
                        }
                        
                        value += blockValue;
                    }
                }
            }
        }
        
        // Para fichas rojas, bloquear fichas azules que avanzan hacia la meta roja
        if (move.piece.color === 'red') {
            const bluePieces = boardAnalysis.bluePieces;
            
            for (let bluePos of bluePieces) {
                // Verificar si la ficha azul está avanzando hacia la meta roja
                if (bluePos.row >= 5) { // Fichas azules en la mitad inferior
                    // Calcular si nuestro movimiento bloquea una ruta de avance
                    const blocksRoute = this.blocksAdvanceRoute(move.to, bluePos, 'blue');
                    
                    if (blocksRoute) {
                        // Valor por bloquear, mayor si la ficha está más cerca de la meta
                        let blockValue = 10; // Valor base
                        
                        // Bonus por distancia a la meta
                        blockValue += (bluePos.row - 5) * 2; // Fila 9 = +8, Fila 8 = +6, etc.
                        
                        // Bonus si bloqueamos en el centro
                        if (move.to.col >= 3 && move.to.col <= 5) {
                            blockValue += 2;
                        }
                        
                        value += blockValue;
                    }
                }
            }
        }
        
        return value;
    },
    
    blocksAdvanceRoute: function(blockPosition, enemyPosition, enemyColor) {
        // Verificar si nuestra posición bloquea una ruta de avance del enemigo
        const goalRow = enemyColor === 'blue' ? 0 : BOARD_ROWS - 1;
        
        // Verificar si nuestro bloqueo está en el camino
        if (enemyColor === 'blue') {
            // Ficha azul va hacia arriba (row disminuye)
            return blockPosition.row < enemyPosition.row && 
                   Math.abs(blockPosition.col - enemyPosition.col) <= 1;
        } else {
            // Ficha roja va hacia abajo (row aumenta)
            return blockPosition.row > enemyPosition.row && 
                   Math.abs(blockPosition.col - enemyPosition.col) <= 1;
        }
    },
    
    analyzeBlockingMoves: function(blockMoves, boardAnalysis) {
        const scoredMoves = blockMoves.map(move => {
            let score = move.blockValue || 0;
            
            // Bonus por bloqueos que no nos ponen en peligro
            if (!this.isPositionInDanger(move.to)) {
                score += 3;
            }
            
            // Penalización por ponernos en peligro
            if (this.isPositionInDanger(move.to)) {
                score -= 2;
            }
            
            return { move, score };
        });
        
        scoredMoves.sort((a, b) => b.score - a.score);
        return scoredMoves[0].move;
    }
};

// Función para limpiar selecciones de fichas que ya no existen
function clearInvalidSelection() {
    if (gameState.selectedPiece) {
        const selectedCell = gameState.board[gameState.selectedPiece.row][gameState.selectedPiece.col];
        
        // Si la ficha seleccionada ya no existe, está eliminándose, o no es del jugador actual
        if (!selectedCell.piece || 
            selectedCell.piece.eliminating || 
            selectedCell.piece.color !== gameState.currentPlayer) {
            
            gameState.selectedPiece = null;
            gameState.showingHints = false;
            gameState.hintMoves = [];
            
            // Actualizar la interfaz para quitar las sugerencias
            createBoardHTML();
        }
    }
}

// Función para limpiar fichas en estado inconsistente (eliminando pero no eliminadas)
function cleanupInconsistentPieces() {
    let needsUpdate = false;
    
    // Solo limpiar fichas que han estado eliminando por más de 500ms
    // Esto permite que las animaciones normales se completen
    const now = Date.now();
    
    for (let r = 0; r < BOARD_ROWS; r++) {
        for (let c = 0; c < BOARD_COLS; c++) {
            const cell = gameState.board[r][c];
            if (cell.piece && cell.piece.eliminating) {
                // Si no tiene timestamp, agregarlo
                if (!cell.piece.eliminatingStartTime) {
                    cell.piece.eliminatingStartTime = now;
                }
                // Si ha estado eliminando por más de 500ms, limpiarla
                else if (now - cell.piece.eliminatingStartTime > 500) {
                    gameState.board[r][c].piece = null;
                    needsUpdate = true;
                }
            }
        }
    }
    
    if (needsUpdate) {
        createBoardHTML();
        updateGameInfo();
    }
}

// Ejecutar limpieza automática cada 2 segundos
setInterval(cleanupInconsistentPieces, 2000);

// Movimiento de la CPU (rojo) con sistema de dificultad
function cpuMove() {
    // Si el juego ha terminado, no hacer movimientos
    if (gameState.gameEnded) {
        return;
    }
    
    // Limpiar selección de ficha eliminada antes del turno de la CPU
    clearInvalidSelection();
    
    // Si el jugador está cerca de la victoria, verificar si ya ganó antes de mover
    if (isPlayerNearVictory()) {
        checkGameEnd();
        // Si el juego terminó por victoria del jugador, no continuar con el movimiento de la CPU
        if (gameState.gameEnded) {
            return;
        }
    }
    
    // Recopilar todas las fichas rojas y sus movimientos posibles
    const availablePieces = [];
    
    for (let r = 0; r < BOARD_ROWS; r++) {
        for (let c = 0; c < BOARD_COLS; c++) {
            const cell = gameState.board[r][c];
            const piece = cell.piece;
            // Solo considerar fichas rojas que estén vivas (no eliminándose)
            if (piece && piece.color === 'red' && !piece.eliminating) {
                const possibleMoves = getPossibleMoves({ row: r, col: c }, 'red');
                const validMoves = possibleMoves.filter(move => 
                    isValidMove({ row: r, col: c }, move, 'red')
                );
                
                if (validMoves.length > 0) {
                    availablePieces.push({
                        from: { row: r, col: c },
                        piece: piece,
                        moves: validMoves
                    });
                }
            }
        }
    }
    
    // Si no hay fichas que puedan moverse, pasar turno
    if (availablePieces.length === 0) {
                    gameState.currentPlayer = 'blue';
                    gameState.turnNumber += 1;
                    updateGameInfo();
                    return;
                }
    
    // Usar la IA según el nivel de dificultad seleccionado
    let selectedMove;
    switch (gameSettings.cpuDifficulty) {
        case 'beginner':
            selectedMove = cpuAI.beginner(availablePieces);
            break;
        case 'intermediate':
            selectedMove = cpuAI.intermediate(availablePieces);
            break;
        case 'expert':
            selectedMove = cpuAI.expert(availablePieces);
            break;
        default:
            selectedMove = cpuAI.beginner(availablePieces);
    }
    
    // Ejecutar el movimiento seleccionado
    const from = selectedMove.from;
    const to = selectedMove.to;
    
    // Verificar si está entrando a la meta roja (fila 10)
    if (to.row === RED_GOAL_ROW) {
        // Ficha llega a la meta - eliminar del tablero y aumentar contador
        gameState.board[from.row][from.col].piece = null;
        gameState.redArrived += 1;
        gameState.redPieces -= 1;
        gameState.redPoints += 2; // Puntos por llegar a la meta
        audioManager.playRivalGoalWarning();
        
        // Actualizar interfaz
            createBoardHTML();
            updateGameInfo();
            
        // Cambiar turno al jugador
        gameState.currentPlayer = 'blue';
        gameState.turnNumber += 1;
        updateGameInfo();
        return;
    }
    // Verificar si está entrando a la meta azul (fila 0)
    else if (to.row === BLUE_GOAL_ROW) {
        // Ficha llega a la meta - eliminar del tablero y aumentar contador
        gameState.board[from.row][from.col].piece = null;
        gameState.redArrived += 1;
        gameState.redPieces -= 1;
        gameState.redPoints += 2; // Puntos por llegar a la meta
        audioManager.playRivalGoalWarning();
        
        // Actualizar interfaz
        createBoardHTML();
        updateGameInfo();
        
        // Cambiar turno al jugador
        gameState.currentPlayer = 'blue';
        gameState.turnNumber += 1;
        updateGameInfo();
        return;
    } else {
        // Verificar si hay eliminación
        const toCell = gameState.board[to.row][to.col];
        if (toCell.piece && toCell.piece.color !== selectedMove.piece.color) {
            // Actualizar contadores inmediatamente cuando la CPU mata una ficha
            if (selectedMove.piece.color === 'red') {
                    // Roja elimina azul
                    gameState.bluePieces -= 1;
                    gameState.redEliminated += 1;
                    gameState.redPoints += 1;
                } else {
                    // Azul elimina roja
                    gameState.redPieces -= 1;
                    gameState.blueEliminated += 1;
                    gameState.bluePoints += 1;
                }
                
            // Crear ficha eliminada con animación de muerte
            const eliminatingPiece = { ...toCell.piece, eliminating: true, eliminatingStartTime: Date.now() };
            gameState.board[to.row][to.col].piece = eliminatingPiece;
            
            // Actualizar interfaz para mostrar animación de muerte
            createBoardHTML();
            updateGameInfo();
            
            // Después de la animación de muerte, colocar ficha atacante
            setTimeout(() => {
                // Colocar ficha atacante en la casilla (sin animación)
                const finalPiece = { ...selectedMove.piece };
                gameState.board[to.row][to.col].piece = finalPiece;
                gameState.board[from.row][from.col].piece = null;
                
                // Limpiar cualquier selección que pueda estar en la ficha eliminada
                if (gameState.selectedPiece && 
                    gameState.selectedPiece.row === to.row && 
                    gameState.selectedPiece.col === to.col) {
                    gameState.selectedPiece = null;
                    gameState.showingHints = false;
                    gameState.hintMoves = [];
                }
                
                // Actualizar interfaz
                createBoardHTML();
                updateGameInfo();
                
                // Cambiar turno al jugador después de completar el movimiento
                gameState.currentPlayer = 'blue';
                gameState.turnNumber += 1;
                updateGameInfo();
            }, 400); // Duración de la animación de muerte
            
            // Determinar quién está eliminando para el sonido apropiado
            const isPlayerEliminating = selectedMove.piece.color === 'blue';
            audioManager.playElimination(isPlayerEliminating);
        } else {
            // Movimiento normal con animación
            const movingPiece = { ...selectedMove.piece, moving: true };
            gameState.board[to.row][to.col].piece = movingPiece;
            gameState.board[from.row][from.col].piece = null;
            audioManager.playPieceMove();
            
            // Actualizar interfaz inmediatamente para mostrar la animación
            createBoardHTML();
            updateGameInfo();
            
            // Quitar la animación después de que termine
            setTimeout(() => {
                if (gameState.board[to.row][to.col].piece) {
                    gameState.board[to.row][to.col].piece.moving = false;
                    createBoardHTML();
                }
            }, 500);
    }
    
        // Cambiar turno al jugador (solo para movimientos normales)
    gameState.currentPlayer = 'blue';
    gameState.turnNumber += 1;
    updateGameInfo();
    }
}

// Función para formatear el tiempo
function formatTime(milliseconds) {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// Función para actualizar el tiempo de partida
function updateGameTime() {
    if (gameState.gameStartTime && !gameState.gameEnded) {
        const currentTime = Date.now();
        const elapsedTime = currentTime - gameState.gameStartTime;
        const gameTimeElement = document.getElementById('gameTime');
        if (gameTimeElement) {
            gameTimeElement.textContent = formatTime(elapsedTime);
        }
    }
}

function updateGameInfo() {
    // Elementos de estadísticas del jugador rojo
    const redPieces = document.getElementById('redPieces');
    const redEliminated = document.getElementById('redEliminated');
    const redArrived = document.getElementById('redArrived');
    const redPoints = document.getElementById('redPoints');
    const redIndicator = document.getElementById('redIndicator');
    
    // Elementos de estadísticas del jugador azul
    const bluePieces = document.getElementById('bluePieces');
    const blueEliminated = document.getElementById('blueEliminated');
    const blueArrived = document.getElementById('blueArrived');
    const bluePoints = document.getElementById('bluePoints');
    const blueIndicator = document.getElementById('blueIndicator');
    
    // Contador de turno
    const turnNumber = document.getElementById('turnNumber');
    
    // Actualizar indicadores de jugador activo
    if (redIndicator) {
        redIndicator.className = gameState.currentPlayer === 'red' ? 'player-indicator red-indicator active' : 'player-indicator red-indicator';
    }
    if (blueIndicator) {
        blueIndicator.className = gameState.currentPlayer === 'blue' ? 'player-indicator blue-indicator active' : 'player-indicator blue-indicator';
    }
    
    // Actualizar contador de turno
    if (turnNumber) {
        turnNumber.textContent = gameState.turnNumber;
    }
    
    // Actualizar tiempo de partida
    updateGameTime();
    
    // Actualizar estadísticas del jugador rojo
    if (redPieces) redPieces.textContent = gameState.redPieces;
    if (redEliminated) redEliminated.textContent = gameState.redEliminated;
    if (redArrived) redArrived.textContent = gameState.redArrived;
    if (redPoints) redPoints.textContent = gameState.redPoints;
    
    // Actualizar estadísticas del jugador azul
    if (bluePieces) bluePieces.textContent = gameState.bluePieces;
    if (blueEliminated) blueEliminated.textContent = gameState.blueEliminated;
    if (blueArrived) blueArrived.textContent = gameState.blueArrived;
    if (bluePoints) bluePoints.textContent = gameState.bluePoints;
    
    // Actualizar nombre del jugador
    const playerNameElement = document.getElementById('playerName');
    if (playerNameElement) {
        playerNameElement.textContent = gameSettings.playerName;
    }
    
    // Verificar condiciones de fin de partida
    checkGameEnd();
}

// Función para verificar si el jugador está cerca de ganar
function isPlayerNearVictory() {
    const threshold = Math.min(VICTORY_CHECK_THRESHOLD, POINTS_TO_WIN - 1);
    return gameState.bluePoints >= (POINTS_TO_WIN - threshold);
}

// Función para actualizar el indicador de dificultad de la CPU
function updateDifficultyIndicator() {
    const difficultyIndicator = document.getElementById('cpuDifficultyIndicator');
    if (!difficultyIndicator) return;
    
    const textElement = difficultyIndicator.querySelector('.difficulty-text');
    
    switch (gameSettings.cpuDifficulty) {
        case 'beginner':
            textElement.textContent = 'Principiante';
            break;
        case 'intermediate':
            textElement.textContent = 'Intermedio';
            break;
        case 'expert':
            textElement.textContent = 'Experto';
            break;
        default:
            textElement.textContent = 'Principiante';
    }
}

// Función para mostrar texto de victoria/derrota
function showVictoryText(winner, message) {
    // Eliminar cualquier overlay existente
    const existingOverlay = document.querySelector('.victory-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    // Evitar crear múltiples overlays si ya existe uno
    if (document.querySelector('.victory-overlay')) {
        return;
    }
    
    // Crear el overlay
    const overlay = document.createElement('div');
    overlay.className = 'victory-overlay';
    
    // Crear el texto
    const textElement = document.createElement('div');
    textElement.className = `victory-text ${winner === 'blue' ? 'victory' : 'defeat'}`;
    
    // Determinar el texto a mostrar
    if (winner === 'blue') {
        textElement.textContent = '¡VICTORIA!';
    } else {
        textElement.textContent = 'DERROTA';
    }
    
    overlay.appendChild(textElement);
    document.body.appendChild(overlay);
    
    // Activar la animación después de un pequeño delay para que se renderice
    setTimeout(() => {
        textElement.classList.add('show');
    }, 50);
}

// Función para ocultar el texto de victoria/derrota
function hideVictoryText() {
    const overlay = document.querySelector('.victory-overlay');
    if (overlay) {
        overlay.remove();
    }
}

// Función para verificar si hay animaciones de eliminación en curso
function hasEliminatingAnimations() {
    for (let r = 0; r < BOARD_ROWS; r++) {
        for (let c = 0; c < BOARD_COLS; c++) {
            const cell = gameState.board[r][c];
            if (cell.piece && cell.piece.eliminating) {
                return true;
            }
        }
    }
    return false;
}

// Función para verificar condiciones de fin de partida
function checkGameEnd() {
    // Verificar si algún equipo se quedó sin fichas
    if (gameState.redPieces <= 0) {
        endGame('blue', `${gameSettings.playerName} ha ganado`);
        return;
    }
    
    if (gameState.bluePieces <= 0) {
        endGame('red', 'CPU ha ganado');
        return;
    }
    
    // Verificar si algún equipo alcanzó los puntos necesarios para ganar
    if (gameState.redPoints >= POINTS_TO_WIN) {
        endGame('red', 'CPU ha ganado');
        return;
    }
    
    if (gameState.bluePoints >= POINTS_TO_WIN) {
        endGame('blue', `${gameSettings.playerName} ha ganado`);
        return;
    }
    
}

// Función para finalizar el juego
function endGame(winner, message) {
    // Evitar ejecución múltiple si el juego ya terminó
    if (gameState.gameEnded) {
        return;
    }
    
    // Detener el juego
    gameState.gameEnded = true;
    
    // Calcular tiempo total de la partida
    if (gameState.gameStartTime) {
        gameState.gameEndTime = Date.now();
        gameState.gameDuration = gameState.gameEndTime - gameState.gameStartTime;
    }
    
    // Limpiar intervalo de tiempo
    if (gameState.timeInterval) {
        clearInterval(gameState.timeInterval);
        gameState.timeInterval = null;
    }
    
    // Verificar si hay animaciones de eliminación en curso
    const hasAnimations = hasEliminatingAnimations();
    const animationDelay = hasAnimations ? 600 : 0; // Esperar 600ms si hay animaciones
    
    // Mostrar texto de victoria/derrota con delay si hay animaciones
    setTimeout(() => {
        showVictoryText(winner, message);
        
        // Reproducir sonido apropiado según el ganador
        if (winner === 'blue') {
            // Jugador gana - delay para que coincida con la animación del texto
            setTimeout(() => {
                audioManager.playVictory();
            }, 600);
        } else {
            // CPU gana - delay pequeño para sincronización
            setTimeout(() => {
                audioManager.playDefeat();
            }, 300);
        }
        
        // Mostrar modal de resumen después de que termine el sonido
        setTimeout(() => {
            hideVictoryText(); // Ocultar el texto antes de mostrar el resumen
            showGameSummary(winner, message);
        }, winner === 'blue' ? 2000 : 1500);
        
    }, animationDelay);
}

// Función para mostrar el resumen del juego
function showGameSummary(winner, message) {
    // Eliminar cualquier modal existente antes de crear uno nuevo
    const existingModal = document.querySelector('.game-summary-modal');
    if (existingModal) {
        existingModal.remove();
    }
    
    // Eliminar también cualquier overlay de victoria que pueda quedar
    const existingOverlay = document.querySelector('.victory-overlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    const modal = document.createElement('div');
    modal.className = 'game-summary-modal';
    
    let winnerName, winnerClass;
    
    winnerName = winner === 'red' ? 'CPU' : gameSettings.playerName;
    winnerClass = winner === 'red' ? 'winner-red' : 'winner-blue';
    
    // Determinar quién gana en cada estadística
    const pointsWinner = gameState.redPoints > gameState.bluePoints ? 'red' : 
                        gameState.bluePoints > gameState.redPoints ? 'blue' : 'tie';
    const eliminatedWinner = gameState.redEliminated > gameState.blueEliminated ? 'red' : 
                            gameState.blueEliminated > gameState.redEliminated ? 'blue' : 'tie';
    const arrivedWinner = gameState.redArrived > gameState.blueArrived ? 'red' : 
                         gameState.blueArrived > gameState.redArrived ? 'blue' : 'tie';
    
    modal.innerHTML = `
        <div class="game-summary-container">
            <div class="boards-container">
                <div class="board" id="finalGameBoard">
                    <!-- El tablero final se generará aquí -->
                </div>
                <div class="right-panel">
                    <div class="stats-panel">
                        <h1 class="summary-title ${winnerClass}" style="font-size: 2.2em; margin-bottom: 25px; text-align: center;">${message}</h1>
                        
                        <div class="stats-container">
                            <div class="stat-row">
                                <div class="stat-value red ${pointsWinner === 'red' ? 'winner' : ''}">${gameState.redPoints}</div>
                                <div class="stat-label-center">Puntos</div>
                                <div class="stat-value blue ${pointsWinner === 'blue' ? 'winner' : ''}">${gameState.bluePoints}</div>
                            </div>
                            
                            <div class="stat-row">
                                <div class="stat-value red ${eliminatedWinner === 'red' ? 'winner' : ''}">${gameState.redEliminated}</div>
                                <div class="stat-label-center">Eliminadas</div>
                                <div class="stat-value blue ${eliminatedWinner === 'blue' ? 'winner' : ''}">${gameState.blueEliminated}</div>
                            </div>
                            
                            <div class="stat-row">
                                <div class="stat-value red ${arrivedWinner === 'red' ? 'winner' : ''}">${gameState.redArrived}</div>
                                <div class="stat-label-center">Completadas</div>
                                <div class="stat-value blue ${arrivedWinner === 'blue' ? 'winner' : ''}">${gameState.blueArrived}</div>
                            </div>
                        </div>
                        
                        <div class="turns-played" style="margin-top: 15px;">
                            <div class="turns-label">Turnos Jugados</div>
                            <div class="turns-value">${gameState.turnNumber}</div>
                            <div class="separator">|</div>
                            <div class="time-label">Tiempo Total</div>
                            <div class="time-value">${formatTime(gameState.gameDuration)}</div>
                        </div>
                    </div>
                    
                    <div class="actions-panel">
                        <button class="game-btn primary" onclick="startNewGame()">
                            <span class="btn-text">Nueva Partida</span>
                            <div class="btn-glow"></div>
                        </button>
                        <button class="game-btn secondary" onclick="goToMenu()">
                            <span class="btn-text">Volver al Menú</span>
                            <div class="btn-glow"></div>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Aplicar el tema correcto a los paneles de estadísticas y crear el tablero final
    setTimeout(() => {
        // Crear el tablero final
        createFinalBoard();
        
        const statsPanel = document.querySelector('.stats-panel');
        const actionsPanel = document.querySelector('.actions-panel');
        const turnsPlayed = document.querySelector('.turns-played');
        
    if (gameSettings.theme === 'light') {
            if (statsPanel) {
                statsPanel.classList.add('light-theme');
                statsPanel.classList.remove('dark-theme');
            }
            if (actionsPanel) {
                actionsPanel.classList.add('light-theme');
                actionsPanel.classList.remove('dark-theme');
            }
            if (turnsPlayed) {
                turnsPlayed.classList.add('light-theme');
                turnsPlayed.classList.remove('dark-theme');
            }
    } else {
            if (statsPanel) {
                statsPanel.classList.add('dark-theme');
                statsPanel.classList.remove('light-theme');
            }
            if (actionsPanel) {
                actionsPanel.classList.add('dark-theme');
                actionsPanel.classList.remove('light-theme');
            }
            if (turnsPlayed) {
                turnsPlayed.classList.add('dark-theme');
                turnsPlayed.classList.remove('light-theme');
            }
        }
    }, 100);
}

// Función para crear el tablero final
function createFinalBoard() {
    // Asegurar que el tablero del juego esté actualizado antes de crear el resumen
    createBoardHTML();
    
    const boardElement = document.getElementById('finalGameBoard');
    boardElement.innerHTML = '';
    
    // Aplicar el tema correcto al tablero final
    if (gameSettings.theme === 'light') {
        boardElement.classList.add('light-theme');
        boardElement.classList.remove('dark-theme');
        } else {
        boardElement.classList.add('dark-theme');
        boardElement.classList.remove('light-theme');
    }
    
    // Crear las filas del tablero
    for (let row = 0; row < BOARD_ROWS; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'board-row';
        
        // Para las filas de meta, crear una sola columna que ocupe todo el ancho
        if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
            const cell = gameState.board[row][0]; // Solo usamos la primera celda como referencia
            const cellElement = document.createElement('div');
            
            // Clases CSS para la celda de meta
            let className = `board-cell ${cell.type} goal-row`;
            
            cellElement.className = className;
            cellElement.dataset.row = row;
            cellElement.dataset.col = 'all';
            
            rowElement.appendChild(cellElement);
        } else {
            // Para el resto de filas, crear las 9 columnas normales
            for (let col = 0; col < BOARD_COLS; col++) {
                const cell = gameState.board[row][col];
                const cellElement = document.createElement('div');
                
                // Clases CSS para la celda
                cellElement.className = `board-cell ${cell.type}`;
                cellElement.dataset.row = row;
                cellElement.dataset.col = col;
                
                // Si hay una ficha en esta celda
                if (cell.piece) {
                    const pieceElement = document.createElement('div');
                    let pieceClass = `piece ${cell.piece.color}`;
                    
                    pieceElement.className = pieceClass;
                    pieceElement.dataset.pieceId = cell.piece.id;
                    cellElement.appendChild(pieceElement);
                }
                
                rowElement.appendChild(cellElement);
            }
        }
        
        boardElement.appendChild(rowElement);
    }
}

// Función para empezar una nueva partida
function startNewGame() {
    audioManager.playButtonClick();
    // Remover el modal
    const modal = document.querySelector('.game-summary-modal');
    if (modal) {
        modal.remove();
    }
    
    // Iniciar nueva partida
    startGame();
}

// Función para volver al menú
function goToMenu() {
    audioManager.playButtonClick();
    // Remover el modal
    const modal = document.querySelector('.game-summary-modal');
    if (modal) {
        modal.remove();
    }
    
    // Volver al menú principal
    showScreen(document.getElementById('startScreen'));
}



// Funciones de tema

function applyTheme() {
    const gameContainer = document.querySelector('.game-container');
    
    if (gameSettings.theme === 'light') {
        gameContainer.classList.add('light-theme');
        gameContainer.classList.remove('dark-theme');
    } else {
        gameContainer.classList.remove('light-theme');
        gameContainer.classList.add('dark-theme');
    }
}


// Elementos del DOM
const startScreen = document.getElementById('startScreen');
const optionsScreen = document.getElementById('optionsScreen');
const gameScreen = document.getElementById('gameScreen');
const difficultyModal = document.getElementById('difficultyModal');

const playBtn = document.getElementById('playBtn');
const optionsBtn = document.getElementById('optionsBtn');
const exitBtn = document.getElementById('exitBtn');
const backBtn = document.getElementById('backBtn');
const menuBtn = document.getElementById('menuBtn');
const cancelDifficulty = document.getElementById('cancelDifficulty');

const themeToggle = document.querySelector('.theme-toggle');
const themeOptions = document.querySelectorAll('.theme-option');
const soundEffectsCheckbox = document.getElementById('soundEffects');
const playerNameInput = document.getElementById('playerNameInput');

// Efectos de sonido (simulados)
function playSound(soundType) {
    if (!gameSettings.soundEffects) return;
    
    // Aquí puedes agregar sonidos reales más tarde
    console.log(`Reproduciendo sonido: ${soundType}`);
    
    // Los efectos visuales se han eliminado para evitar destellos
}

// Funciones de navegación
function showScreen(screenToShow) {
    // Ocultar todas las pantallas
    startScreen.classList.add('hidden');
    optionsScreen.classList.add('hidden');
    gameScreen.classList.add('hidden');
    
    // Mostrar la pantalla seleccionada
    screenToShow.classList.remove('hidden');
    
    // Reproducir sonido de navegación
    playSound('navigate');
}

function startGame() {
    console.log('Mostrando modal de dificultad');
    showDifficultyModal();
}

function showDifficultyModal() {
    // Ocultar pantalla de inicio
    startScreen.classList.add('hidden');
    
    // Mostrar modal de dificultad
    difficultyModal.classList.remove('hidden');
    
    // Configurar opciones de dificultad
    setupDifficultyOptions();
}

function setupDifficultyOptions() {
    // Configurar selección de tablero
    const boardOptions = document.querySelectorAll('.board-option');
    
    // Remover selección previa de tableros
    boardOptions.forEach(option => {
        option.classList.remove('selected');
    });
    
    // Seleccionar el tablero actual
    const currentBoardOption = document.querySelector(`[data-size="${gameSettings.boardSize}"]`);
    if (currentBoardOption) {
        currentBoardOption.classList.add('selected');
    }
    
    // Agregar event listeners para selección de tablero
    boardOptions.forEach(option => {
        option.addEventListener('click', function() {
            audioManager.playButtonClick();
            
            // Remover selección de todas las opciones de tablero
            boardOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Seleccionar la opción clickeada
            this.classList.add('selected');
            
            // Guardar el tamaño de tablero seleccionado
            gameSettings.boardSize = this.dataset.size;
        });
    });
    
    // Configurar selección de dificultad
    const difficultyOptions = document.querySelectorAll('.difficulty-option');
    
    // Remover selección previa de dificultad
    difficultyOptions.forEach(option => {
        option.classList.remove('selected');
    });
    
    // Seleccionar la dificultad actual
    const currentOption = document.querySelector(`[data-level="${gameSettings.cpuDifficulty}"]`);
    if (currentOption) {
        currentOption.classList.add('selected');
    }
    
    // Agregar event listeners para dificultad
    difficultyOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remover selección de todas las opciones
            difficultyOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Seleccionar la opción clickeada
            this.classList.add('selected');
            
            // Guardar la dificultad seleccionada
            gameSettings.cpuDifficulty = this.dataset.level;
            
            // Iniciar el juego con la configuración seleccionada
            setTimeout(() => {
                initializeGame();
            }, 300);
        });
    });
}

function hideDifficultyModal() {
    difficultyModal.classList.add('hidden');
}

function initializeGame() {
    console.log('Iniciando juego con configuración:', gameSettings);
    
    // Ocultar modal de dificultad
    hideDifficultyModal();
    
    // Configurar el tablero según el tamaño seleccionado
    configureBoard();
    
    // Inicializar el estado del juego
    gameState.currentPlayer = 'blue'; // Empieza el jugador humano (azul)
    gameState.selectedPiece = null;
    
    // Inicializar estadísticas
    gameState.redEliminated = 0;
    gameState.blueEliminated = 0;
    gameState.redArrived = 0;
    gameState.blueArrived = 0;
    gameState.redPoints = 0;
    gameState.bluePoints = 0;
    gameState.turnNumber = 1;
    
    // Inicializar tiempo de partida
    gameState.gameStartTime = Date.now();
    gameState.gameEndTime = null;
    gameState.gameDuration = 0;
    
    // Reiniciar estado del juego
    gameState.gameEnded = false;
    gameState.showingHints = false;
    gameState.hintMoves = [];
    
    // Crear el tablero
    initializeBoard();
    
    // Mostrar la pantalla de juego
    showScreen(gameScreen);
    
    // Actualizar indicador de dificultad
    updateDifficultyIndicator();
    
    // Crear el HTML del tablero
    createBoardHTML();
    
    // Mostrar información de formaciones
    showFormationInfo();
    
    // Actualizar la información del juego
    updateGameInfo();
    
    // Iniciar actualización del tiempo cada segundo
    if (gameState.timeInterval) {
        clearInterval(gameState.timeInterval);
    }
    gameState.timeInterval = setInterval(updateGameTime, 1000);
    
    // Aplicar tema y actualizar botón
    applyTheme();
    
    playSound('start');
    console.log('Tablero inicializado:', gameState.board);
}

function showOptions() {
    showScreen(optionsScreen);
    playSound('menu');
}

function backToMenu() {
    showScreen(startScreen);
    playSound('back');
}

function exitGame() {
    playSound('exit');
    
    // Crear efecto de fade out
    document.body.style.transition = 'opacity 1s ease-out';
    document.body.style.opacity = '0';
    
    setTimeout(() => {
        // En una aplicación real, esto cerraría la ventana
        // Para web, podemos mostrar un mensaje o redirigir
        if (confirm('¿Estás seguro de que quieres salir del juego?')) {
            window.close(); // Esto solo funciona si la ventana fue abierta por JavaScript
            // Alternativa: window.location.href = 'about:blank';
        } else {
            // Restaurar la opacidad si el usuario cancela
            document.body.style.opacity = '1';
        }
    }, 1000);
}

// Funciones de configuración

function updateTheme() {
    const activeOption = document.querySelector('.theme-option.active');
    if (activeOption) {
        gameSettings.theme = activeOption.dataset.theme;
        applyTheme();
        updateThemeToggle();
        saveSettings(); // Guardar automáticamente
        console.log('Tema actualizado:', gameSettings.theme);
    }
}

function updateThemeToggle() {
    themeOptions.forEach(option => {
        option.classList.remove('active');
        if (option.dataset.theme === gameSettings.theme) {
            option.classList.add('active');
        }
    });
    
    // Actualizar el atributo data-active del contenedor
    if (themeToggle) {
        themeToggle.setAttribute('data-active', gameSettings.theme);
    }
}

function updateSoundEffects() {
    gameSettings.soundEffects = soundEffectsCheckbox.checked;
    console.log('Efectos de sonido:', gameSettings.soundEffects ? 'activados' : 'desactivados');
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Botones principales
    playBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        startGame();
    });
    optionsBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        showOptions();
    });
    exitBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        exitGame();
    });
    backBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        backToMenu();
    });
    cancelDifficulty.addEventListener('click', function() {
        audioManager.playButtonClick();
        hideDifficultyModal();
        startScreen.classList.remove('hidden');
    });
    menuBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        backToMenu();
    });
    
    // Botón temporal para finalizar partida (solo para pruebas)
    const endGameBtn = document.getElementById('endGameBtn');
    if (endGameBtn) {
        endGameBtn.addEventListener('click', function() {
            audioManager.playButtonClick();
            endGame('blue', 'Final Manual');
        });
    }
    
    
    // Controles de opciones
    // Event listeners para el selector de tema
    themeOptions.forEach(option => {
        option.addEventListener('click', function() {
            audioManager.playButtonClick();
            // Remover active de todas las opciones
            themeOptions.forEach(opt => opt.classList.remove('active'));
            // Agregar active a la opción clickeada
            this.classList.add('active');
            updateTheme();
        });
    });
    soundEffectsCheckbox.addEventListener('change', updateSoundEffects);
    
    // Efectos de hover para botones
    const allButtons = document.querySelectorAll('.game-btn');
    allButtons.forEach(button => {
        button.addEventListener('mouseenter', () => {
            playSound('hover');
        });
    });
    
    // Atajos de teclado
    document.addEventListener('keydown', function(event) {
        switch(event.key) {
            case 'Enter':
                if (!startScreen.classList.contains('hidden')) {
                    startGame();
                }
                break;
            case 'Escape':
                if (!optionsScreen.classList.contains('hidden')) {
                    backToMenu();
                } else if (!gameScreen.classList.contains('hidden')) {
                    backToMenu();
                }
                break;
            case '1':
                if (!startScreen.classList.contains('hidden')) {
                    startGame();
                }
                break;
            case '2':
                if (!startScreen.classList.contains('hidden')) {
                    showOptions();
                }
                break;
            case '3':
                if (!startScreen.classList.contains('hidden')) {
                    exitGame();
                }
                break;
        }
    });
    
    console.log('Juego inicializado correctamente');
    console.log('Atajos de teclado:');
    console.log('- Enter o 1: Jugar');
    console.log('- 2: Opciones');
    console.log('- 3: Salir');
    console.log('- Escape: Volver al menú');
});

// Función para guardar configuración (localStorage)
function saveSettings() {
    localStorage.setItem('gameSettings', JSON.stringify(gameSettings));
    console.log('Configuración guardada');
}

// Función para cargar configuración
function loadSettings() {
    const saved = localStorage.getItem('gameSettings');
    if (saved) {
        gameSettings = { ...gameSettings, ...JSON.parse(saved) };
        
        // Aplicar configuración a los controles
        updateThemeToggle();
        soundEffectsCheckbox.checked = gameSettings.soundEffects;
        if (playerNameInput) {
            playerNameInput.value = gameSettings.playerName;
        }
        
        // Aplicar tema si estamos en la pantalla de juego
        applyTheme();
        
        // Actualizar nombre del jugador en la interfaz
        updateGameInfo();
        
        console.log('Configuración cargada:', gameSettings);
    }
}

// Cargar configuración al iniciar
window.addEventListener('load', function() {
    loadSettings();
    audioManager.init();
});

// Guardar configuración cuando cambie
// Los event listeners del tema ya manejan el guardado
soundEffectsCheckbox.addEventListener('change', saveSettings);

// Guardar nombre del jugador
if (playerNameInput) {
    playerNameInput.addEventListener('input', function() {
        gameSettings.playerName = this.value.trim() || 'Jugador';
        saveSettings();
        updateGameInfo(); // Actualizar inmediatamente en la interfaz
    });
}

// El tema se guarda automáticamente en toggleTheme()

// Animación adicional para las estrellas
function createRandomStars() {
    const backgroundAnimation = document.querySelector('.background-animation');
    
    setInterval(() => {
        const star = document.createElement('div');
        star.className = 'star';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animationDelay = Math.random() * 3 + 's';
        
        backgroundAnimation.appendChild(star);
        
        // Remover la estrella después de la animación
        setTimeout(() => {
            if (star.parentNode) {
                star.parentNode.removeChild(star);
            }
        }, 3000);
    }, 2000);
}

// Iniciar animación de estrellas
setTimeout(createRandomStars, 1000);