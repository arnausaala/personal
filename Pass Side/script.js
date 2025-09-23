// Variables globales
let gameSettings = {
    soundEffects: true,
    theme: 'dark', // 'dark' o 'light'
    playerName: 'Jugador',
    cpuDifficulty: 'beginner', // 'beginner', 'intermediate', 'expert'
    boardSize: 'classic' // 'bala', 'classic', 'marathon'
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
    bala: [
        { name: "Cadena Alterna", pattern: "oxoxo", weight: 0.25 }, 
        { name: "Centro Puro", pattern: "xooox", weight: 0.25 },
        { name: "Lateral Izquierdo", pattern: "oooxx", weight: 0.10 },    
        { name: "Lateral Derecho", pattern: "xxooo", weight: 0.10 },   
        { name: "Distribución Aleatoria", pattern: "random", weight: 0.30 }
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
    marathon: [
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
    
    // Obtener patrones predefinidos para evitar duplicados
    const patterns = BOARD_PATTERNS[gameSettings.boardSize];
    const predefinedPatterns = patterns.filter(p => p.pattern !== "random").map(p => p.pattern);
    
    // Calcular el número total de combinaciones posibles
    const totalCombinations = factorial(numCols) / (factorial(numPieces) * factorial(numCols - numPieces));
    
    // Calcular cuántas formaciones aleatorias son realmente posibles (excluyendo predefinidas)
    const availableRandomCombinations = totalCombinations - predefinedPatterns.length;
    
    // Calcular el peso individual de cada distribución aleatoria
    const randomDist = patterns.find(d => d.pattern === "random");
    const individualWeight = randomDist ? randomDist.weight / availableRandomCombinations : 0.01;
    
    let pattern;
    let attempts = 0;
    const maxAttempts = 100; // Evitar bucle infinito
    
    do {
        const positions = Array.from({length: numCols}, (_, i) => i);
        const selectedPositions = [];
        
        // Seleccionar el número correcto de posiciones aleatorias
        while (selectedPositions.length < numPieces) {
            const randomIndex = Math.floor(Math.random() * positions.length);
            const selectedPosition = positions.splice(randomIndex, 1)[0];
            selectedPositions.push(selectedPosition);
        }
        
        // Crear el patrón
        pattern = "x".repeat(numCols); // Espacios vacíos
        for (let pos of selectedPositions) {
            pattern = pattern.substring(0, pos) + "o" + pattern.substring(pos + 1);
        }
        
        attempts++;
    } while (predefinedPatterns.includes(pattern) && attempts < maxAttempts);
    
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
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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
    bala: { rows: 7, cols: 5, pieces: 3, points: 3 },
    classic: { rows: 11, cols: 9, pieces: 7, points: 7 },
    marathon: { rows: 15, cols: 11, pieces: 8, points: 8 }
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
    if (gameSettings.boardSize === 'bala') {
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
    if (gameSettings.boardSize === 'bala') {
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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
    
    // Agregar evento click para mostrar/ocultar modal
    cpuFormationDiv.addEventListener('click', () => {
        toggleFormationModal(gameState.redDistribution, 'CPU', 'red');
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
    
    // Agregar evento click para mostrar/ocultar modal
    playerFormationDiv.addEventListener('click', () => {
        toggleFormationModal(gameState.blueDistribution, 'Tu Formación', 'blue');
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

// Función para alternar (mostrar/ocultar) modal de formación
function toggleFormationModal(distribution, title, teamColor) {
    // Determinar el lado según el equipo (CPU=izquierda, Jugador=derecha)
    const sideClass = teamColor === 'red' ? 'formation-panel-left' : 'formation-panel-right';
    
    // Verificar si ya existe un panel para este equipo
    const existingPanel = document.querySelector(`.formation-side-panel.${sideClass}`);
    
    if (existingPanel) {
        // Si existe, ocultarlo
        hideFormationModal({ target: existingPanel.querySelector('.formation-panel-close') });
    } else {
        // Si no existe, mostrarlo
        showFormationModal(distribution, title, teamColor);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para mostrar información detallada de formación en panel lateral
function showFormationModal(distribution, title, teamColor) {
    // Detectar el tema actual
    const isDarkMode = document.body.classList.contains('dark-theme');
    const themeClass = isDarkMode ? 'dark-theme' : 'light-theme';
    
    // Determinar el lado según el equipo (CPU=izquierda, Jugador=derecha)
    const sideClass = teamColor === 'red' ? 'formation-panel-left' : 'formation-panel-right';
    
    // Remover panel existente si existe para evitar duplicados
    const existingPanel = document.querySelector(`.formation-side-panel.${sideClass}`);
    if (existingPanel) {
        existingPanel.remove();
    }
    
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
    let panel;
    
    if (event.target && event.target.closest) {
        // Si se llama desde un botón de cerrar
        const closeBtn = event.target;
        panel = closeBtn.closest('.formation-side-panel');
    } else {
        // Si se llama directamente desde toggleFormationModal
        panel = event.target;
    }
    
    if (panel) {
        panel.classList.add('hide');
        setTimeout(() => {
            panel.remove();
        }, 300);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para limpiar todos los modales de formaciones
function clearAllFormationModals() {
    const allModals = document.querySelectorAll('.formation-side-panel');
    allModals.forEach(modal => {
        modal.classList.add('hide');
        setTimeout(() => {
            modal.remove();
        }, 300);
    });
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

    // Reglas específicas para el modo Bala (5 filas de juego: 2 propio + 1 seguro + 2 contrario)
    if (gameSettings.boardSize === 'bala') {
    if (color === 'blue') {
            // En modo Bala: fila de aparición azul (fila 5, índice 5) y siguiente fila (fila 4, índice 4)
            // Solo movimientos de 1 casilla: recto o hacia los lados
            if (from.row === BLUE_START_ROW || from.row === BLUE_START_ROW - 1) {
                const forward = dRow === -1 && dCol === 0;
                const left = dRow === 0 && dCol === -1;
                const right = dRow === 0 && dCol === 1;
                return forward || left || right;
            }
            // Zona segura (fila 3, índice 3): movimientos recto o diagonal hacia adelante
            else if (from.row === 3) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario (filas 0-2, índices 0-2): movimientos normales
            else if (from.row >= 0 && from.row <= 2) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta azul (fila 0, índice 0): no se puede mover desde aquí
            else if (from.row === BLUE_GOAL_ROW) {
                return false;
            }
        } else { // color === 'red'
            // En modo Bala: fila de aparición roja (fila 1, índice 1) y siguiente fila (fila 2, índice 2)
            // Solo movimientos de 1 casilla: recto o hacia los lados
            if (from.row === RED_START_ROW || from.row === RED_START_ROW + 1) {
                const forward = dRow === 1 && dCol === 0;
                const left = dRow === 0 && dCol === -1;
                const right = dRow === 0 && dCol === 1;
                return forward || left || right;
            }
            // Zona segura (fila 3, índice 3): movimientos recto o diagonal hacia adelante
            else if (from.row === 3) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario (filas 4-6, índices 4-6): movimientos normales
            else if (from.row >= 4 && from.row <= 6) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta roja (fila 6, índice 6): no se puede mover desde aquí
            else if (from.row === RED_GOAL_ROW) {
                return false;
            }
        }
        return false; // Si no coincide con ninguna regla del modo Bala
    }

    // Reglas específicas para modo Classic (9 filas de juego: 4 propio + 1 seguro + 4 contrario)
    if (gameSettings.boardSize === 'classic') {
        if (color === 'blue') {
            // Campo propio azul: filas 6-9 (4 filas: 6,7,8,9)
            if (from.row >= 6 && from.row <= 9) {
                // En la fila de aparición (fila 9): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 5): 1 adelante o diagonal adelante
            else if (from.row === 5) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario: filas 1-4 (4 filas: 1,2,3,4): 1 adelante o diagonal adelante
            else if (from.row >= 1 && from.row <= 4) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta azul (fila 0): no se puede mover desde aquí
            else if (from.row === BLUE_GOAL_ROW) {
                return false;
            }
            // Permitir movimiento a la meta azul desde fila de aparición roja
            else if (to.row === BLUE_GOAL_ROW) {
                return from.row === RED_START_ROW;
            }
        } else { // color === 'red'
            // Campo propio rojo: filas 1-4 (4 filas: 1,2,3,4)
            if (from.row >= 1 && from.row <= 4) {
                // En la fila de aparición (fila 1): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 5): 1 adelante o diagonal adelante
            else if (from.row === 5) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario: filas 6-9 (4 filas: 6,7,8,9): 1 adelante o diagonal adelante
            else if (from.row >= 6 && from.row <= 9) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta roja (fila 10): no se puede mover desde aquí
            else if (from.row === RED_GOAL_ROW) {
                return false;
            }
            // Permitir movimiento a la meta roja desde fila de aparición azul
            else if (to.row === RED_GOAL_ROW) {
                return from.row === BLUE_START_ROW;
            }
        }
        return false; // Si no coincide con ninguna regla del modo Classic
    }

    // Reglas para modo Marathon (13 filas de juego: 6 propio + 1 seguro + 6 contrario)
    if (gameSettings.boardSize === 'marathon') {
        if (color === 'blue') {
            // Campo propio azul: filas 8-13 (índices 8-13)
            if (from.row >= 8 && from.row <= 13) {
                // En la fila de aparición (fila 13): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 7): 1 adelante o diagonal adelante
            else if (from.row === 7) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario: filas 0-6 (índices 0-6): 1 adelante o diagonal adelante
            else if (from.row >= 0 && from.row <= 6) {
                const forward = dRow === -1 && dCol === 0;
                const diagonalLeft = dRow === -1 && dCol === -1;
                const diagonalRight = dRow === -1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta azul (fila 0): no se puede mover desde aquí
            else if (from.row === BLUE_GOAL_ROW) {
                return false;
            }
            // Permitir movimiento a la meta azul desde fila de aparición roja
            else if (to.row === BLUE_GOAL_ROW) {
                return from.row === RED_START_ROW;
            }
        } else { // color === 'red'
            // Campo propio rojo: filas 1-6 (índices 1-6)
            if (from.row >= 1 && from.row <= 6) {
                // En la fila de aparición (fila 1): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 7): 1 adelante o diagonal adelante
            else if (from.row === 7) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Campo contrario: filas 8-14 (índices 8-14): 1 adelante o diagonal adelante
            else if (from.row >= 8 && from.row <= 14) {
                const forward = dRow === 1 && dCol === 0;
                const diagonalLeft = dRow === 1 && dCol === -1;
                const diagonalRight = dRow === 1 && dCol === 1;
                return forward || diagonalLeft || diagonalRight;
            }
            // Meta roja (fila 14): no se puede mover desde aquí
            else if (from.row === RED_GOAL_ROW) {
                return false;
            }
            // Permitir movimiento a la meta roja desde fila de aparición azul
            else if (to.row === RED_GOAL_ROW) {
                return from.row === BLUE_START_ROW;
            }
        }
        return false; // Si no coincide con ninguna regla del modo Marathon
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
    
    // Reglas específicas para el modo Bala (5 filas de juego: 2 propio + 1 seguro + 2 contrario)
    if (gameSettings.boardSize === 'bala') {
        // No se puede eliminar en la zona segura (fila 3, índice 3)
        if (to.row === 3) return false;
        
        // Solo se puede eliminar si el atacante está en su campo propio
    if (color === 'blue') {
            // Ficha azul solo puede eliminar si está en su campo propio (filas 4-6)
            return from.row >= 4 && from.row <= 6;
    } else {
            // Ficha roja solo puede eliminar si está en su campo propio (filas 0-2)
            return from.row >= 0 && from.row <= 2;
        }
    } else if (gameSettings.boardSize === 'classic') {
        // Reglas específicas para modo Classic (9 filas de juego: 4 propio + 1 seguro + 4 contrario)
        // No se puede eliminar en la zona segura (fila 5)
        if (to.row === 5) return false;
        
        // Solo se puede eliminar si el atacante está en su campo propio
        if (color === 'blue') {
            // Ficha azul solo puede eliminar si está en su campo propio (filas 6-9)
            const canEliminate = from.row >= 6 && from.row <= 9;
            return canEliminate;
        } else {
            // Ficha roja solo puede eliminar si está en su campo propio (filas 1-4)
            const canEliminate = from.row >= 1 && from.row <= 4;
            return canEliminate;
        }
    } else if (gameSettings.boardSize === 'marathon') {
        // Reglas específicas para modo Marathon (15 filas)
        // No se puede eliminar en la zona segura (fila 7)
        if (to.row === 7) return false;
        
        // Solo se puede eliminar si el atacante está en su campo propio
        if (color === 'blue') {
            // Ficha azul solo puede eliminar si está en su campo propio (filas 8-13)
            const canEliminate = from.row >= 8 && from.row <= 13;
            return canEliminate;
        } else {
            // Ficha roja solo puede eliminar si está en su campo propio (filas 1-6)
            const canEliminate = from.row >= 1 && from.row <= 6;
            return canEliminate;
        }
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

    // Reglas específicas para el modo Bala (5 filas de juego: 2 propio + 1 seguro + 2 contrario)
    if (gameSettings.boardSize === 'bala') {
    if (color === 'blue') {
            // En modo Bala: fila de aparición azul (fila 5, índice 5) y siguiente fila (fila 4, índice 4)
            // Solo movimientos de 1 casilla: recto o hacia los lados
            if (row === BLUE_START_ROW || row === BLUE_START_ROW - 1) {
                // Solo movimientos de 1 casilla
                moves.push({ row: row - 1, col: col }); // Adelante
                moves.push({ row: row, col: col - 1 }); // Izquierda
                moves.push({ row: row, col: col + 1 }); // Derecha
            }
            // Zona segura (fila 3, índice 3): movimientos recto o diagonal hacia adelante
            else if (row === 3) {
                moves.push({ row: row - 1, col: col }); // Adelante
                moves.push({ row: row - 1, col: col - 1 }); // Diagonal izquierda
                moves.push({ row: row - 1, col: col + 1 }); // Diagonal derecha
            }
            // Campo contrario (filas 0-2, índices 0-2): movimientos normales
            else if (row >= 0 && row <= 2) {
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 1, col: col - 1 });
                moves.push({ row: row - 1, col: col + 1 });
            }
            // Meta azul (fila 0, índice 0): no se puede mover desde aquí
            else if (row === BLUE_GOAL_ROW) {
                // No se puede mover desde la meta
            }
        } else { // color === 'red'
            // En modo Bala: fila de aparición roja (fila 1, índice 1) y siguiente fila (fila 2, índice 2)
            // Solo movimientos de 1 casilla: recto o hacia los lados
            if (row === RED_START_ROW || row === RED_START_ROW + 1) {
                // Solo movimientos de 1 casilla
                moves.push({ row: row + 1, col: col }); // Adelante
                moves.push({ row: row, col: col - 1 }); // Izquierda
                moves.push({ row: row, col: col + 1 }); // Derecha
            }
            // Zona segura (fila 3, índice 3): movimientos recto o diagonal hacia adelante
            else if (row === 3) {
                moves.push({ row: row + 1, col: col }); // Adelante
                moves.push({ row: row + 1, col: col - 1 }); // Diagonal izquierda
                moves.push({ row: row + 1, col: col + 1 }); // Diagonal derecha
            }
            // Campo contrario (filas 4-6, índices 4-6): movimientos normales
            else if (row >= 4 && row <= 6) {
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 1, col: col - 1 });
                moves.push({ row: row + 1, col: col + 1 });
            }
            // Meta roja (fila 6, índice 6): no se puede mover desde aquí
            else if (row === RED_GOAL_ROW) {
                // No se puede mover desde la meta
            }
        }
    } else if (gameSettings.boardSize === 'classic') {
        // Reglas específicas para modo Classic (9 filas de juego: 4 propio + 1 seguro + 4 contrario)
        if (color === 'blue') {
            // Campo propio azul: filas 6-9 (4 filas: 6,7,8,9)
            if (row >= 6 && row <= 9) {
                if (row === BLUE_START_ROW) {
                    // Fila de aparición (fila 9): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 5): 1 adelante o diagonal adelante
            else if (row === 5) {
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 1, col: col - 1 });
                moves.push({ row: row - 1, col: col + 1 });
            }
            // Campo contrario: filas 1-4 (4 filas: 1,2,3,4): 1 adelante o diagonal adelante
            else if (row >= 1 && row <= 4) {
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 1, col: col - 1 });
                moves.push({ row: row - 1, col: col + 1 });
            }
            // Meta azul (fila 0): no se puede mover desde aquí
            else if (row === BLUE_GOAL_ROW) {
                // No se puede mover desde la meta
            }
            // Se puede llegar a la meta azul desde la fila de aparición roja
            if (row === RED_START_ROW) {
                moves.push({ row: 0, col: 0 }); // Meta azul
            }
        } else { // color === 'red'
            // Campo propio rojo: filas 1-6 (índices 1-6)
            if (row >= 1 && row <= 6) {
                if (row === RED_START_ROW) {
                    // Fila de aparición (fila 1): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 5): 1 adelante o diagonal adelante
            else if (row === 5) {
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 1, col: col - 1 });
                moves.push({ row: row + 1, col: col + 1 });
            }
            // Campo contrario: filas 6-10 (índices 6-10): 1 adelante o diagonal adelante
            else if (row >= 6 && row <= 10) {
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 1, col: col - 1 });
                moves.push({ row: row + 1, col: col + 1 });
            }
            // Meta roja (fila 10): no se puede mover desde aquí
            else if (row === RED_GOAL_ROW) {
                // No se puede mover desde la meta
            }
            // Se puede llegar a la meta roja desde la fila de aparición azul
            if (row === BLUE_START_ROW) {
                moves.push({ row: 10, col: 0 }); // Meta roja
            }
        }
    } else if (gameSettings.boardSize === 'marathon') {
        // Reglas específicas para modo Marathon (15 filas)
        if (color === 'blue') {
            // Campo propio azul: filas 8-13 (índices 8-13)
            if (row >= 8 && row <= 13) {
                if (row === BLUE_START_ROW) {
                    // Fila de aparición (fila 13): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 7): 1 adelante o diagonal adelante
            else if (row === 7) {
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 1, col: col - 1 });
                moves.push({ row: row - 1, col: col + 1 });
            }
            // Campo contrario: filas 0-6 (índices 0-6): 1 adelante o diagonal adelante
            else if (row >= 0 && row <= 6) {
                moves.push({ row: row - 1, col: col });
                moves.push({ row: row - 1, col: col - 1 });
                moves.push({ row: row - 1, col: col + 1 });
            }
            // Meta azul (fila 0): no se puede mover desde aquí
            else if (row === BLUE_GOAL_ROW) {
                // No se puede mover desde la meta
            }
            // Se puede llegar a la meta azul desde la fila de aparición roja
            if (row === RED_START_ROW) {
                moves.push({ row: 0, col: 0 }); // Meta azul
            }
        } else { // color === 'red'
            // Campo propio rojo: filas 1-6 (índices 1-6)
            if (row >= 1 && row <= 6) {
                if (row === RED_START_ROW) {
                    // Fila de aparición (fila 1): 1 o 2 casillas adelante o 1 o 2 hacia el lado
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
            // Zona segura (fila 7): 1 adelante o diagonal adelante
            else if (row === 7) {
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 1, col: col - 1 });
                moves.push({ row: row + 1, col: col + 1 });
            }
            // Campo contrario: filas 8-14 (índices 8-14): 1 adelante o diagonal adelante
            else if (row >= 8 && row <= 14) {
                moves.push({ row: row + 1, col: col });
                moves.push({ row: row + 1, col: col - 1 });
                moves.push({ row: row + 1, col: col + 1 });
            }
            // Meta roja (fila 14): no se puede mover desde aquí
            else if (row === RED_GOAL_ROW) {
                // No se puede mover desde la meta
            }
            // Se puede llegar a la meta roja desde la fila de aparición azul
            if (row === BLUE_START_ROW) {
                moves.push({ row: 14, col: 0 }); // Meta roja
            }
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
                    // Verificar que la eliminación sea válida según las reglas
                    if (canEliminate(pieceData.from, move, pieceData.piece.color)) {
                    eliminationMoves.push({
                        from: pieceData.from,
                        to: move,
                        piece: pieceData.piece,
                        type: 'elimination'
                    });
                    }
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
            move.row === target.row && move.col === target.col && canEliminate(from, target, color)
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
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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
    
    // Actualizar información del tablero
    const boardSizeDisplay = document.getElementById('boardSizeDisplay');
    const boardPointsDisplay = document.getElementById('boardPointsDisplay');
    
    if (boardSizeDisplay) {
        // Mostrar tamaño del área de juego (sin contar filas de meta)
        boardSizeDisplay.textContent = `${BOARD_ROWS - 2}x${BOARD_COLS}`;
    }
    
    if (boardPointsDisplay) {
        boardPointsDisplay.textContent = `${POINTS_TO_WIN} pts`;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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
        document.body.classList.remove('dark-theme');
    } else {
        gameContainer.classList.remove('light-theme');
        gameContainer.classList.add('dark-theme');
        document.body.classList.add('dark-theme');
    }
    
    // Actualizar imágenes de movimientos si estamos en esa pantalla
    updateMovementImagesTheme();
}


// Elementos del DOM
const startScreen = document.getElementById('startScreen');
const tutorialScreen = document.getElementById('tutorialScreen');
const optionsScreen = document.getElementById('optionsScreen');
const gameScreen = document.getElementById('gameScreen');
const difficultyModal = document.getElementById('difficultyModal');

const playBtn = document.getElementById('playBtn');
const tutorialBtn = document.getElementById('tutorialBtn');
const optionsBtn = document.getElementById('optionsBtn');
const exitBtn = document.getElementById('exitBtn');
const backBtn = document.getElementById('backBtn');
const menuBtn = document.getElementById('menuBtn');
const cancelDifficulty = document.getElementById('cancelDifficulty');

// Elementos del tutorial
const interactiveTutorial = document.getElementById('interactiveTutorial');
const guideTutorial = document.getElementById('guideTutorial');
const backFromTutorial = document.getElementById('backFromTutorial');

// Elementos del tutorial interactivo
const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
const tutorialBoard = document.getElementById('tutorialBoard');
const navPrev = document.getElementById('navPrev');
const navNext = document.getElementById('navNext');
const backFromInteractiveTutorial = document.getElementById('backFromInteractiveTutorial');

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
    tutorialScreen.classList.add('hidden');
    interactiveTutorialScreen.classList.add('hidden');
    optionsScreen.classList.add('hidden');
    gameScreen.classList.add('hidden');
    
    // Mostrar la pantalla seleccionada
    screenToShow.classList.remove('hidden');
    
    // Reproducir sonido de navegación
    playSound('navigate');
}

function startGame() {
    clearAllFormationModals();
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
    
    // Mostrar texto por defecto (sin selección visual)
    showDefaultPreview();
    
    // Actualizar información de la partida
    updateGameInfoPreview();
    
    // Aplicar efectos de dificultad
    applyDifficultyToPreviewBoard();
    
    // Verificar si se puede habilitar el botón de empezar
    checkCanStartGame();
}

function setupDifficultyOptions() {
    // Configurar selección de tablero
    const boardOptions = document.querySelectorAll('.board-option');
    
    // Remover selección previa de tableros
    boardOptions.forEach(option => {
        option.classList.remove('selected');
        // Remover event listeners anteriores si existen
        option.replaceWith(option.cloneNode(true));
    });
    
    // Obtener referencias actualizadas después de clonar
    const freshBoardOptions = document.querySelectorAll('.board-option');
    
    // No seleccionar nada por defecto
    gameSettings.boardSize = '';
    
    // Agregar event listeners para selección de tablero
    freshBoardOptions.forEach(option => {
        option.addEventListener('click', function() {
            audioManager.playButtonClick();
            
            // Verificar si ya está seleccionada
            if (this.classList.contains('selected')) {
                // Deseleccionar
                this.classList.remove('selected');
                gameSettings.boardSize = '';
                
                // Mostrar texto por defecto
                showDefaultPreview();
            } else {
                // Remover selección de todas las opciones de tablero
                freshBoardOptions.forEach(opt => opt.classList.remove('selected'));
                
                // Seleccionar la opción clickeada
                this.classList.add('selected');
                
                // Guardar el tamaño de tablero seleccionado
                gameSettings.boardSize = this.dataset.size;
                
                // Generar el tablero de previsualización según el modo seleccionado
                if (this.dataset.size === 'bala') {
                    createIndependentBalaBoard();
                } else if (this.dataset.size === 'classic') {
                    createIndependentClassicBoard();
                } else if (this.dataset.size === 'marathon') {
                    createIndependentMarathonBoard();
                }
            }
            
            // Actualizar información de la partida
            updateGameInfoPreview();
            
            // Aplicar efectos de dificultad
            applyDifficultyToPreviewBoard();
            
            // Verificar si se puede habilitar el botón de empezar
            checkCanStartGame();
        });
    });
    
    // Configurar selección de dificultad
    const difficultyOptions = document.querySelectorAll('.difficulty-option');
    
    // Remover selección previa de dificultad
    difficultyOptions.forEach(option => {
        option.classList.remove('selected');
        // Remover event listeners anteriores si existen
        option.replaceWith(option.cloneNode(true));
    });
    
    // Obtener referencias actualizadas después de clonar
    const freshDifficultyOptions = document.querySelectorAll('.difficulty-option');
    
    // No seleccionar nada por defecto
    gameSettings.cpuDifficulty = '';
    
    // Agregar event listeners para dificultad
    freshDifficultyOptions.forEach(option => {
        option.addEventListener('click', function() {
            audioManager.playButtonClick();
            
            // Verificar si ya está seleccionada
            if (this.classList.contains('selected')) {
                // Deseleccionar
                this.classList.remove('selected');
                gameSettings.cpuDifficulty = '';
            } else {
            // Remover selección de todas las opciones
                freshDifficultyOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Seleccionar la opción clickeada
            this.classList.add('selected');
            
            // Guardar la dificultad seleccionada
            gameSettings.cpuDifficulty = this.dataset.level;
            }
            
            // Aplicar efectos de dificultad
            applyDifficultyToPreviewBoard();
            
            // Verificar si se puede habilitar el botón de empezar
            checkCanStartGame();
        });
    });
}

function hideDifficultyModal() {
    difficultyModal.classList.add('hidden');
}

function initializeGame() {
    console.log('Iniciando juego con configuración:', gameSettings);
    
    // Limpiar modales de formaciones
    clearAllFormationModals();
    
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
    clearAllFormationModals();
    showScreen(optionsScreen);
    playSound('menu');
}

function backToMenu() {
    clearAllFormationModals();
    showScreen(startScreen);
    playSound('back');
}

// Funciones del tutorial
function showTutorial() {
    clearAllFormationModals();
    showScreen(tutorialScreen);
    playSound('menu');
}

function startInteractiveTutorial() {
    clearAllFormationModals();
    showScreen(interactiveTutorialScreen);
    initializeInteractiveTutorial();
    playSound('menu');
}

// Variables del tutorial interactivo
let currentTutorialStep = 1;
    const totalTutorialSteps = 4;

function initializeInteractiveTutorial() {
    currentTutorialStep = 1;
    
    // Ocultar el tablero inmediatamente
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (tutorialBoard) {
        tutorialBoard.style.display = 'none';
    }
    
    // Ejecutar inmediatamente sin delay
    updateTutorialStep();
}

function createTutorialBoard() {
    tutorialBoard.innerHTML = '';
    
    // Configuración del tablero Clásico (basado en createIndependentClassicBoard)
    const rows = 11;
    const cols = 9;
    const cellSize = 35; // Tamaño más pequeño para el tutorial
    
    // Configurar el contenedor del tablero
    tutorialBoard.className = 'board classic-preview';
    tutorialBoard.style.display = 'flex';
    tutorialBoard.style.flexDirection = 'column';
    tutorialBoard.style.gap = '0';
    tutorialBoard.style.background = 'var(--board-bg)';
    tutorialBoard.style.padding = '10px';
    tutorialBoard.style.borderRadius = '8px';
    tutorialBoard.style.boxShadow = '0 4px 16px rgba(0, 0, 0, 0.2)';
    tutorialBoard.style.border = '2px solid var(--board-border)';
    tutorialBoard.style.position = 'relative';
    tutorialBoard.style.transform = 'scale(0.8)';
    tutorialBoard.style.transformOrigin = 'center';
    tutorialBoard.style.pointerEvents = 'none';
    
    // Calcular dimensiones de las filas de meta
    const goalRowWidth = cellSize * cols;
    const goalRowHeight = 25;
    
    // Crear las filas del tablero
    for (let row = 0; row < rows; row++) {
        const boardRow = document.createElement('div');
        boardRow.className = 'board-row';
        boardRow.style.display = 'flex';
        boardRow.style.gap = '0';
        
        // Determinar el tipo de fila
        let rowType = 'normal';
        if (row === 0) {
            rowType = 'blue-goal';
        } else if (row === rows - 1) {
            rowType = 'red-goal';
        }
        
        if (rowType === 'blue-goal' || rowType === 'red-goal') {
            // Crear fila de meta
            const goalCell = document.createElement('div');
            goalCell.className = `board-cell ${rowType} goal-row`;
            goalCell.style.width = `${goalRowWidth}px`;
            goalCell.style.height = `${goalRowHeight}px`;
            goalCell.style.display = 'flex';
            goalCell.style.alignItems = 'center';
            goalCell.style.justifyContent = 'center';
            goalCell.style.border = 'none';
            goalCell.style.cursor = 'default';
            goalCell.style.pointerEvents = 'none';
            
            // Aplicar estilos de meta
            if (rowType === 'blue-goal') {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-blue-primary) 0deg 90deg,
                        var(--goal-blue-secondary) 90deg 180deg,
                        var(--goal-blue-primary) 180deg 270deg,
                        var(--goal-blue-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '10px 10px';
            } else {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-red-primary) 0deg 90deg,
                        var(--goal-red-secondary) 90deg 180deg,
                        var(--goal-red-primary) 180deg 270deg,
                        var(--goal-red-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '10px 10px';
            }
            
            boardRow.appendChild(goalCell);
        } else {
            // Crear filas normales
            for (let col = 0; col < cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.cursor = 'default';
                cell.style.pointerEvents = 'none';
                cell.style.borderRight = '1px solid var(--board-grid-line)';
                cell.style.borderBottom = '1px solid var(--board-grid-line)';
                
                // Determinar el tipo de celda
                let cellType = 'neutral';
                if (row === 1) {
                    cellType = 'red-start';
                } else if (row === rows - 2) {
                    cellType = 'blue-start';
                } else if (row === Math.floor(rows / 2)) {
                    cellType = 'safe-zone';
                } else if (row < Math.floor(rows / 2)) {
                    cellType = 'neutral';
                } else {
                    cellType = 'neutral2';
                }
                
                // Aplicar estilos según el tipo de celda
                switch (cellType) {
                    case 'red-start':
                        cell.style.backgroundColor = 'var(--cell-red-start-bg)';
                        cell.style.border = '2px solid var(--cell-red-start-border)';
                        break;
                    case 'blue-start':
                        cell.style.backgroundColor = 'var(--cell-blue-start-bg)';
                        cell.style.border = '2px solid var(--cell-blue-start-border)';
                        break;
                    case 'safe-zone':
                        cell.style.backgroundColor = 'var(--cell-safe-bg)';
                        cell.style.border = '2px solid var(--cell-safe-border)';
                        break;
                    default:
                        cell.style.backgroundColor = 'var(--cell-neutral-bg)';
                        cell.style.border = '1px solid var(--cell-neutral-border)';
                }
                
                // Agregar fichas de ejemplo
                if (row === 1 && col >= 2 && col <= 6) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = 'piece red';
                    pieceElement.style.width = '20px';
                    pieceElement.style.height = '20px';
                    pieceElement.style.borderRadius = '50%';
                    pieceElement.style.backgroundColor = 'var(--piece-red-color)';
                    pieceElement.style.border = '2px solid var(--piece-red-border)';
                    cell.appendChild(pieceElement);
                } else if (row === rows - 2 && col >= 2 && col <= 6) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = 'piece blue';
                    pieceElement.style.width = '20px';
                    pieceElement.style.height = '20px';
                    pieceElement.style.borderRadius = '50%';
                    pieceElement.style.backgroundColor = 'var(--piece-blue-color)';
                    pieceElement.style.border = '2px solid var(--piece-blue-border)';
                    cell.appendChild(pieceElement);
                }
                
                // Remover bordes de los últimos elementos
                if (col === cols - 1) {
                    cell.style.borderRight = 'none';
                }
                
                boardRow.appendChild(cell);
            }
        }
        
        tutorialBoard.appendChild(boardRow);
    }
    
    // Remover el borde inferior de la última fila
    const lastRow = tutorialBoard.lastElementChild;
    if (lastRow && lastRow.lastElementChild) {
        lastRow.lastElementChild.style.borderBottom = 'none';
    }
}

function updateTutorialStep() {
    // Ocultar todos los pasos
    document.querySelectorAll('.explanation-step').forEach(step => {
        step.classList.remove('active');
    });
    
    // Mostrar el paso actual
    const currentStepElement = document.getElementById(`step${currentTutorialStep}`);
    if (currentStepElement) {
        currentStepElement.classList.add('active');
    }
    
    // Añadir/quitar clase para centrar texto
    const tutorialContent = document.querySelector('.tutorial-content');
    const tutorialTitle = document.getElementById('tutorialTitle');
    const tutorialSubtitle = document.getElementById('tutorialSubtitle');
    
    if (currentTutorialStep === 1) {
        tutorialContent.classList.add('text-only');
        tutorialContent.classList.remove('movements-only');
        tutorialTitle.textContent = 'OBJETIVO';
        tutorialTitle.removeAttribute('data-step');
        tutorialSubtitle.style.display = 'none';
    } else if (currentTutorialStep === 2) {
        tutorialContent.classList.remove('text-only', 'movements-only');
        tutorialTitle.textContent = 'TABLERO';
        tutorialTitle.setAttribute('data-step', '2');
        tutorialSubtitle.style.display = 'none';
    } else if (currentTutorialStep === 3) {
        tutorialContent.classList.remove('text-only');
        tutorialContent.classList.add('movements-only');
        tutorialTitle.textContent = 'MOVIMIENTOS';
        tutorialTitle.setAttribute('data-step', '3');
        tutorialSubtitle.style.display = 'none';
    } else if (currentTutorialStep === 4) {
        tutorialContent.classList.remove('text-only');
        tutorialContent.classList.add('movements-only');
        tutorialTitle.textContent = 'PUNTUACIÓN';
        tutorialTitle.setAttribute('data-step', '4');
        tutorialSubtitle.style.display = 'none';
    } else {
        tutorialContent.classList.remove('text-only', 'movements-only');
        tutorialTitle.textContent = 'TUTORIAL INTERACTIVO';
        tutorialTitle.removeAttribute('data-step');
        tutorialSubtitle.style.display = 'block';
    }
    
    // Mostrar u ocultar el tablero según el paso
    if (currentTutorialStep === 1) {
        // En el paso 1 (resumen), ocultar el tablero
        tutorialBoard.style.display = 'none';
    } else if (currentTutorialStep === 2) {
        // En el paso 2 (tablero), mostrar el tablero clásico en el contenedor del tablero
        tutorialBoard.style.display = 'block';
        createTutorialClassicBoard();
        // Inicializar pestañas después de crear el tablero
        setTimeout(() => {
            initializeTutorialTabs();
        }, 100);
    } else if (currentTutorialStep === 3) {
        // En el paso 3 (movimientos), ocultar el tablero y inicializar pestañas
        tutorialBoard.style.display = 'none';
        setTimeout(() => {
            initializeMovementTabs();
        }, 100);
    } else if (currentTutorialStep === 4) {
        // En el paso 4 (puntuación), ocultar el tablero e inicializar pestañas de puntuación
        tutorialBoard.style.display = 'none';
        setTimeout(() => {
            initializePuntuacionTabs();
        }, 100);
    } else {
        // En los demás pasos, mostrar el tablero interactivo
        tutorialBoard.style.display = 'block';
            createTutorialBoard();
    }
    
    // Actualizar botones de navegación (comentado - se maneja en updateNavigationButtons)
    // prevStep.disabled = currentTutorialStep === 1;
    
    // if (currentTutorialStep === totalTutorialSteps) {
    //     nextStep.innerHTML = '<span class="btn-text">FINALIZAR</span><div class="btn-glow"></div>';
    // } else {
    //     nextStep.innerHTML = '<span class="btn-text">SIGUIENTE</span><div class="btn-glow"></div>';
    // }
    
    // Aplicar efectos visuales al tablero según el paso (solo si el tablero está visible)
    if (currentTutorialStep > 2) {
        highlightTutorialBoard();
    }
    
    // Actualizar navegación
    updateNavigationButtons();
}

// Función para actualizar los botones de navegación
function updateNavigationButtons() {
    const navPrev = document.getElementById('navPrev');
    const navNext = document.getElementById('navNext');
    const stepButtons = document.querySelectorAll('.tutorial-step-btn');
    
    if (stepButtons.length === 0) {
        return;
    }
    
    // Actualizar estado de flechas
    navPrev.disabled = currentTutorialStep === 1;
    navNext.disabled = currentTutorialStep === totalTutorialSteps;
    
    // Actualizar botones de pasos
    stepButtons.forEach(button => {
        const step = parseInt(button.dataset.step);
        const icon = button.querySelector('.step-icon');
        
        if (step === currentTutorialStep) {
            button.classList.add('active');
            // Forzar el estilo directamente con JavaScript
            if (icon) {
                icon.style.filter = 'brightness(2.2) saturate(2.0) hue-rotate(60deg)';
                icon.style.transform = '';
            }
        } else {
            button.classList.remove('active');
            // Resetear el estilo
            if (icon) {
                icon.style.filter = '';
                icon.style.transform = '';
            }
        }
    });
}

function highlightTutorialBoard() {
    // Remover todas las clases de highlight
    document.querySelectorAll('.board-cell').forEach(cell => {
        cell.style.boxShadow = '';
        cell.style.border = '';
    });
    
    // Aplicar highlight según el paso actual
    const cells = document.querySelectorAll('.board-cell');
    
    switch (currentTutorialStep) {
        case 1: // Objetivo
            // Highlight metas
            cells.forEach(cell => {
                const row = parseInt(cell.dataset.row);
                if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
                    cell.style.boxShadow = '0 0 10px #00d4ff';
                    cell.style.border = '2px solid #00d4ff';
                }
            });
            break;
        case 2: // Campo propio
            cells.forEach(cell => {
                const row = parseInt(cell.dataset.row);
                const cellType = getCellType(row);
                if (cellType === 'START') {
                    cell.style.boxShadow = '0 0 10px #28a745';
                    cell.style.border = '2px solid #28a745';
                }
            });
            break;
        case 3: // Zona segura
            cells.forEach(cell => {
                const row = parseInt(cell.dataset.row);
                const cellType = getCellType(row);
                if (cellType === 'SAFE') {
                    cell.style.boxShadow = '0 0 10px #ffd700';
                    cell.style.border = '2px solid #ffd700';
                }
            });
            break;
        case 4: // Campo contrario
            cells.forEach(cell => {
                const row = parseInt(cell.dataset.row);
                const cellType = getCellType(row);
                if (cellType !== 'BLUE_GOAL' && cellType !== 'RED_GOAL' && cellType !== 'SAFE' && cellType !== 'START') {
                    cell.style.boxShadow = '0 0 10px #dc3545';
                    cell.style.border = '2px solid #dc3545';
                }
            });
            break;
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

function nextTutorialStep() {
    if (currentTutorialStep < totalTutorialSteps) {
        currentTutorialStep++;
        updateTutorialStep();
        playSound('menu');
    } else {
        // Finalizar tutorial
        backFromInteractiveTutorialMenu();
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

function prevTutorialStep() {
    if (currentTutorialStep > 1) {
        currentTutorialStep--;
        updateTutorialStep();
        playSound('menu');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

function backFromInteractiveTutorialMenu() {
    showScreen(tutorialScreen);
    playSound('back');
}

function startGuideTutorial() {
    // TODO: Implementar guía de tutorial
    console.log('Iniciando guía de tutorial');
    playSound('menu');
}

function backFromTutorialMenu() {
    showScreen(startScreen);
    playSound('back');
}

function exitGame() {
    clearAllFormationModals();
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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
    tutorialBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        showTutorial();
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
    
    // Botón para empezar partida desde el modal
    const startGameModalBtn = document.getElementById('startGame');
    if (startGameModalBtn) {
        startGameModalBtn.addEventListener('click', function() {
            if (!this.disabled) {
                audioManager.playButtonClick();
                hideDifficultyModal();
                initializeGame();
            }
        });
    }
    
    menuBtn.addEventListener('click', function() {
        audioManager.playButtonClick();
        backToMenu();
    });
    
    // Event listeners del tutorial
    interactiveTutorial.addEventListener('click', function() {
        audioManager.playButtonClick();
        startInteractiveTutorial();
    });
    guideTutorial.addEventListener('click', function() {
        audioManager.playButtonClick();
        startGuideTutorial();
    });
    backFromTutorial.addEventListener('click', function() {
        console.log('Botón CANCELAR clickeado');
        audioManager.playButtonClick();
        showScreen(startScreen);
        playSound('back');
    });
    
    // Event listeners del tutorial interactivo
    navPrev.addEventListener('click', function() {
        audioManager.playButtonClick();
        prevTutorialStep();
    });
    
    navNext.addEventListener('click', function() {
        audioManager.playButtonClick();
        nextTutorialStep();
    });
    
    // Event listener para finalizar tutorial
    const finishTutorial = document.getElementById('finishTutorial');
    if (finishTutorial) {
        finishTutorial.addEventListener('click', function() {
            audioManager.playButtonClick();
            closeInteractiveTutorial();
        });
    }
    
    // Event listeners para botones de pasos
    const stepButtons = document.querySelectorAll('.tutorial-step-btn');
    stepButtons.forEach(button => {
        button.addEventListener('click', function() {
            audioManager.playButtonClick();
            const step = parseInt(button.dataset.step);
            if (step !== currentTutorialStep) {
                currentTutorialStep = step;
                updateTutorialStep();
            }
        });
    });
    
    backFromInteractiveTutorial.addEventListener('click', function() {
        audioManager.playButtonClick();
        backFromInteractiveTutorialMenu();
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

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
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

// ===== TABLERO DE PREVISUALIZACIÓN INDEPENDIENTE =====

// Función para aplicar efectos de dificultad al tablero de previsualización
function applyDifficultyToPreviewBoard() {
    const previewBoard = document.getElementById('previewBoard');
    if (!previewBoard) return;
    
    // Remover todas las clases de dificultad
    previewBoard.classList.remove('difficulty-beginner', 'difficulty-intermediate', 'difficulty-expert');
    
    // Solo aplicar efectos si hay tanto tablero como dificultad seleccionados
    if (gameSettings.boardSize && gameSettings.cpuDifficulty) {
        const difficultyClasses = {
            'beginner': 'difficulty-beginner',
            'intermediate': 'difficulty-intermediate',
            'expert': 'difficulty-expert'
        };
        
        if (difficultyClasses[gameSettings.cpuDifficulty]) {
            previewBoard.classList.add(difficultyClasses[gameSettings.cpuDifficulty]);
        }
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para actualizar la información de la partida según el modo seleccionado
function updateGameInfoPreview() {
    const piecesInfo = document.getElementById('piecesInfo');
    const formationsInfo = document.getElementById('formationsInfo');
    const pointsInfo = document.getElementById('pointsInfo');
    const separator1 = document.getElementById('separator1');
    const separator2 = document.getElementById('separator2');
    
    if (!piecesInfo || !formationsInfo || !pointsInfo) return;
    
    // Verificar si hay un tablero seleccionado visualmente
    const selectedBoardOption = document.querySelector('.board-option.selected');
    
    if (!selectedBoardOption) {
        // Mostrar mensaje predeterminado ocupando las 3 columnas
        piecesInfo.textContent = '';
        piecesInfo.style.display = 'none';
        formationsInfo.textContent = 'Información sobre la partida';
        formationsInfo.style.display = 'inline';
        formationsInfo.style.gridColumn = '1 / -1'; // Ocupa todas las columnas
        pointsInfo.textContent = '';
        pointsInfo.style.display = 'none';
        return;
    }
    
    // Obtener configuración del tablero seleccionado
    const boardSize = selectedBoardOption.dataset.size;
    const config = BOARD_CONFIGS[boardSize];
    
    if (config) {
        // Mostrar información específica del modo
        piecesInfo.textContent = `${config.pieces} fichas`;
        piecesInfo.style.display = 'inline';
        piecesInfo.style.gridColumn = ''; // Reset
        
        // Calcular formaciones posibles
        const possibleFormations = calculatePossibleFormations(config.pieces, config.cols);
        formationsInfo.textContent = `${possibleFormations} formaciones`;
        formationsInfo.style.display = 'inline';
        formationsInfo.style.gridColumn = ''; // Reset
        
        pointsInfo.textContent = `${config.points} puntos`;
        pointsInfo.style.display = 'inline';
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para calcular formaciones posibles
function calculatePossibleFormations(pieces, cols) {
    // Calcular combinaciones: C(cols, pieces)
    if (pieces > cols) return 0;
    
    let numerator = 1;
    let denominator = 1;
    
    for (let i = 0; i < pieces; i++) {
        numerator *= (cols - i);
        denominator *= (i + 1);
    }
    
    return Math.floor(numerator / denominator);
}

// Función para verificar si se puede empezar el juego
function checkCanStartGame() {
    const startGameBtn = document.getElementById('startGame');
    if (!startGameBtn) return;
    
    // Habilitar solo si hay tanto tablero como dificultad seleccionados
    const hasBoard = gameSettings.boardSize !== '';
    const hasDifficulty = gameSettings.cpuDifficulty !== '';
    
    startGameBtn.disabled = !(hasBoard && hasDifficulty);
}

// Función para mostrar el texto por defecto (sin recuadro)
function showDefaultPreview() {
    const previewBoard = document.getElementById('previewBoard');
    if (!previewBoard) return;
    
    // Limpiar contenido anterior
    previewBoard.innerHTML = '';
    
    // Configurar el contenedor del tablero
    previewBoard.className = '';
    previewBoard.style.display = 'flex';
    previewBoard.style.justifyContent = 'center';
    previewBoard.style.alignItems = 'center';
    previewBoard.style.background = 'transparent';
    previewBoard.style.border = 'none';
    previewBoard.style.boxShadow = 'none';
    previewBoard.style.padding = '0';
    previewBoard.style.borderRadius = '0';
    previewBoard.style.transform = 'translateY(210px)'; // Bajar hasta la mitad de la columna
    previewBoard.style.transformOrigin = '';
    previewBoard.style.position = 'relative';
    previewBoard.style.pointerEvents = 'none';
    
    // Crear elemento de texto
    const textElement = document.createElement('div');
    textElement.textContent = 'Previsualización del tablero';
    textElement.style.color = '#cccccc';
    textElement.style.fontSize = '0.85rem';
    textElement.style.fontWeight = 'normal';
    textElement.style.fontFamily = 'inherit';
    textElement.style.textAlign = 'center';
    textElement.style.pointerEvents = 'none';
    
    previewBoard.appendChild(textElement);
}

// Función para crear el tablero Clásico independiente
function createIndependentClassicBoard() {
    const previewBoard = document.getElementById('previewBoard');
    if (!previewBoard) return;
    
    // Limpiar contenido anterior
    previewBoard.innerHTML = '';
    
    // Configuración del tablero Clásico
    const rows = 11;
    const cols = 9;
    const cellSize = 55; // Tamaño original de celda
    
    // Configurar el contenedor del tablero
    previewBoard.className = 'board classic-preview';
    previewBoard.style.display = 'flex';
    previewBoard.style.flexDirection = 'column';
    previewBoard.style.gap = '0';
    previewBoard.style.background = 'var(--board-bg)';
    previewBoard.style.padding = '20px';
    previewBoard.style.borderRadius = '12px';
    previewBoard.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
    previewBoard.style.border = '3px solid var(--board-border)';
    previewBoard.style.position = 'relative';
    previewBoard.style.transform = ''; // No aplicar transform aquí, se hace en CSS
    previewBoard.style.transformOrigin = 'top center';
    previewBoard.style.pointerEvents = 'none'; // Desactivar interacciones
    
    // Calcular dimensiones de las filas de meta
    const goalRowWidth = cellSize * cols;
    const goalRowHeight = 40;
    
    // Crear las filas del tablero
    for (let row = 0; row < rows; row++) {
        const boardRow = document.createElement('div');
        boardRow.className = 'board-row';
        boardRow.style.display = 'flex';
        boardRow.style.gap = '0';
        
        // Determinar el tipo de fila
        let rowType = 'normal';
        if (row === 0) {
            rowType = 'blue-goal';
        } else if (row === rows - 1) {
            rowType = 'red-goal';
        }
        
        if (rowType === 'blue-goal' || rowType === 'red-goal') {
            // Crear fila de meta
            const goalCell = document.createElement('div');
            goalCell.className = `board-cell ${rowType} goal-row`;
            goalCell.style.width = `${goalRowWidth}px`;
            goalCell.style.height = `${goalRowHeight}px`;
            goalCell.style.display = 'flex';
            goalCell.style.alignItems = 'center';
            goalCell.style.justifyContent = 'center';
            goalCell.style.border = 'none';
            goalCell.style.cursor = 'pointer';
            goalCell.style.pointerEvents = 'auto';
            
            // Aplicar estilos de meta
            if (rowType === 'blue-goal') {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-blue-primary) 0deg 90deg,
                        var(--goal-blue-secondary) 90deg 180deg,
                        var(--goal-blue-primary) 180deg 270deg,
                        var(--goal-blue-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            } else {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-red-primary) 0deg 90deg,
                        var(--goal-red-secondary) 90deg 180deg,
                        var(--goal-red-primary) 180deg 270deg,
                        var(--goal-red-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            }
            
            // Event listeners desactivados - el tablero no es responsivo al cursor
            // goalCell.addEventListener('mouseenter', () => highlightZone(row));
            // goalCell.addEventListener('mouseleave', () => clearHighlight());
            
            boardRow.appendChild(goalCell);
        } else {
            // Crear filas normales
            for (let col = 0; col < cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.cursor = 'default';
                cell.style.pointerEvents = 'none';
                cell.style.borderRight = '1px solid var(--board-grid-line)';
                cell.style.borderBottom = '1px solid var(--board-grid-line)';
                
                // Determinar el tipo de celda
                let cellType = 'neutral';
                if (row === 1) {
                    cellType = 'blue-start';
                } else if (row === rows - 2) {
                    cellType = 'red-start';
                } else if (row === 5) {
                    cellType = 'safe-zone'; // Toda la fila central (fila 5) es zona segura
                }
                
                cell.classList.add(cellType);
                
                // Aplicar estilos según el tipo
                if (cellType === 'neutral' || cellType === 'neutral2') {
                    cell.style.background = 'var(--cell-normal)';
                } else if (cellType === 'blue-start' || cellType === 'red-start') {
                    cell.style.background = 'var(--cell-start)';
                } else if (cellType === 'safe-zone') {
                    cell.style.background = 'var(--cell-safe)';
                    cell.style.position = 'relative';
                    
                    // Crear el indicador interno de zona segura
                    const innerZone = document.createElement('div');
                    innerZone.style.position = 'absolute';
                    innerZone.style.top = '4px';
                    innerZone.style.left = '4px';
                    innerZone.style.right = '4px';
                    innerZone.style.bottom = '4px';
                    innerZone.style.background = 'var(--cell-safe-inner)';
                    innerZone.style.borderRadius = '4px';
                    innerZone.style.border = '2px solid var(--cell-safe-border)';
                    innerZone.style.pointerEvents = 'none';
                    
                    cell.appendChild(innerZone);
                }
                
                boardRow.appendChild(cell);
            }
        }
        
        previewBoard.appendChild(boardRow);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para crear el tablero clásico del tutorial (réplica exacta del tablero de juego)
function createTutorialClassicBoard() {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar contenido anterior
    tutorialBoard.innerHTML = '';
    
    // Configuración del tablero Clásico (igual que el juego real)
    const BOARD_ROWS = 11;
    const BOARD_COLS = 9;
    const BLUE_GOAL_ROW = 0;
    const RED_GOAL_ROW = BOARD_ROWS - 1;
    const RED_START_ROW = 1;
    const BLUE_START_ROW = BOARD_ROWS - 2;
    const SAFE_ZONE_ROW = Math.floor(BOARD_ROWS / 2); // Fila 5 (índice 5)
    
    // Configurar el contenedor del tablero (igual que el juego real)
    tutorialBoard.className = 'board tutorial-classic';
    tutorialBoard.style.display = 'flex';
    tutorialBoard.style.flexDirection = 'column';
    tutorialBoard.style.gap = '0';
    tutorialBoard.style.background = 'var(--board-bg)';
    tutorialBoard.style.padding = '20px';
    tutorialBoard.style.borderRadius = '12px';
    tutorialBoard.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
    tutorialBoard.style.border = '3px solid var(--board-border)';
    tutorialBoard.style.position = 'relative';
    tutorialBoard.style.pointerEvents = 'none'; // Desactivar interacciones
    
    // Función para obtener el tipo de celda (igual que el juego real)
    function getCellType(row) {
        if (row === BLUE_GOAL_ROW) return 'blue-goal';
        if (row === RED_GOAL_ROW) return 'red-goal';
        if (row === RED_START_ROW) return 'red-start';
        if (row === BLUE_START_ROW) return 'blue-start';
        if (row === SAFE_ZONE_ROW) return 'safe-zone';
        
        // Filas neutrales
        if (row < SAFE_ZONE_ROW) {
            return 'neutral';    // Campo del jugador azul
        } else {
            return 'neutral2';   // Campo del jugador rojo
        }
    }
    
    // Crear las filas del tablero (igual que createBoardHTML)
    for (let row = 0; row < BOARD_ROWS; row++) {
        const rowElement = document.createElement('div');
        rowElement.className = 'board-row';
        rowElement.dataset.row = row;
        
        // Para las filas de meta, crear una sola columna que ocupe todo el ancho
        if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
            const cellElement = document.createElement('div');
            const cellType = getCellType(row);
            
            // Clases CSS para la celda de meta (igual que el juego real)
            cellElement.className = `board-cell ${cellType} goal-row`;
            cellElement.dataset.row = row;
            cellElement.dataset.col = 'all';
            
            // Estilos para filas de meta (igual que el juego real)
            cellElement.style.width = 'var(--goal-row-width)';
            cellElement.style.height = 'var(--goal-row-height)';
            cellElement.style.display = 'flex';
            cellElement.style.alignItems = 'center';
            cellElement.style.justifyContent = 'center';
            cellElement.style.border = 'none';
            cellElement.style.pointerEvents = 'none';
            
            rowElement.appendChild(cellElement);
        } else {
            // Para el resto de filas, crear las 9 columnas normales
            for (let col = 0; col < BOARD_COLS; col++) {
                const cellElement = document.createElement('div');
                const cellType = getCellType(row);
                
                // Clases CSS para la celda (igual que el juego real)
                cellElement.className = `board-cell ${cellType}`;
                cellElement.dataset.row = row;
                cellElement.dataset.col = col;
                
                // Estilos de la celda (igual que el juego real)
                cellElement.style.width = 'var(--cell-size)';
                cellElement.style.height = 'var(--cell-size)';
                cellElement.style.display = 'flex';
                cellElement.style.alignItems = 'center';
                cellElement.style.justifyContent = 'center';
                cellElement.style.cursor = 'pointer';
                cellElement.style.transition = 'all 0.3s ease';
                cellElement.style.position = 'relative';
                cellElement.style.borderRight = '1px solid var(--board-grid-line)';
                cellElement.style.borderBottom = '1px solid var(--board-grid-line)';
                cellElement.style.pointerEvents = 'auto';
                
                // Quitar bordes de la última columna y última fila
                if (col === BOARD_COLS - 1) {
                    cellElement.style.borderRight = 'none';
                }
                if (row === BOARD_ROWS - 1) {
                    cellElement.style.borderBottom = 'none';
                }
                
                // Agregar clases de zona para interactividad
                if (row === BLUE_GOAL_ROW) {
                    cellElement.classList.add('zone-blue-goal');
                } else if (row === RED_START_ROW) {
                    cellElement.classList.add('zone-red-start');
                } else if (row < SAFE_ZONE_ROW && row > RED_START_ROW) {
                    cellElement.classList.add('zone-red-field');
                } else if (row === SAFE_ZONE_ROW) {
                    cellElement.classList.add('zone-safe');
                } else if (row > SAFE_ZONE_ROW && row < BLUE_START_ROW) {
                    cellElement.classList.add('zone-blue-field');
                } else if (row === BLUE_START_ROW) {
                    cellElement.classList.add('zone-blue-start');
                } else if (row === RED_GOAL_ROW) {
                    cellElement.classList.add('zone-red-goal');
                }
                
                // Event listeners desactivados - el tablero no es responsivo al cursor
                // cellElement.addEventListener('mouseenter', () => highlightZone(row));
                // cellElement.addEventListener('mouseleave', () => clearHighlight());
                
                rowElement.appendChild(cellElement);
            }
        }
        
        tutorialBoard.appendChild(rowElement);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para resaltar una zona del tablero
function highlightZone(row) {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar timeout anterior si existe
    if (window.highlightTimeout) {
        clearTimeout(window.highlightTimeout);
    }
    
    // Remover resaltado anterior de todas las filas
    tutorialBoard.querySelectorAll('.board-row').forEach(rowElement => {
        rowElement.classList.remove('zone-highlighted');
    });
    
    // Determinar qué filas resaltar según la zona
    const rowsToHighlight = getRowsForZone(row);
    
    // Resaltar todas las filas de la zona
    rowsToHighlight.forEach((rowIndex, index) => {
        const rowElement = tutorialBoard.querySelector(`.board-row[data-row="${rowIndex}"]`);
        if (rowElement) {
            rowElement.classList.add('zone-highlighted');
            
            // Agregar clase field-zone para campos de equipo (múltiples filas)
            if (rowsToHighlight.length > 1) {
                rowElement.classList.add('field-zone');
            }
        }
    });
    
    // Mostrar información de la zona
    showZoneInfo(row);
}

// Función para limpiar el resaltado
function clearHighlight() {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar timeout anterior si existe
    if (window.highlightTimeout) {
        clearTimeout(window.highlightTimeout);
    }
    
    // Pequeño delay para evitar parpadeos
    window.highlightTimeout = setTimeout(() => {
        tutorialBoard.querySelectorAll('.board-row').forEach(rowElement => {
            rowElement.classList.remove('zone-highlighted', 'field-zone');
        });
        
        // Ocultar información de la zona
        hideZoneInfo();
    }, 100);
}

// Función para mostrar información de la zona
function showZoneInfo(row) {
    const zoneInfoElement = document.getElementById('zoneInfo');
    if (!zoneInfoElement) return;
    
    const zoneInfo = getZoneInfo(row);
    if (zoneInfo) {
        // Actualizar el contenido de la zona
        zoneInfoElement.innerHTML = zoneInfo.content;
        
        // Actualizar la pestaña activa
        updateActiveTab(zoneInfo.zone);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para actualizar la pestaña activa
function updateActiveTab(zone) {
    const tabs = document.querySelectorAll('.tutorial-tab');
    tabs.forEach(tab => {
        tab.classList.remove('active');
        if (tab.dataset.zone === zone) {
            tab.classList.add('active');
        }
    });
}

// Función para obtener información detallada de cada zona
function getZoneInfo(row) {
    const BOARD_ROWS = 11;
    const BLUE_GOAL_ROW = 0;
    const RED_GOAL_ROW = BOARD_ROWS - 1;
    const RED_START_ROW = 1;
    const BLUE_START_ROW = BOARD_ROWS - 2;
    const SAFE_ZONE_ROW = Math.floor(BOARD_ROWS / 2);
    
    // Determinar el tipo de zona basado en la fila
    let zoneType;
    if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
        zoneType = 'meta';
    } else if (row === RED_START_ROW || row === BLUE_START_ROW) {
        zoneType = 'aparicion';
    } else if (row < SAFE_ZONE_ROW && row > RED_START_ROW) {
        zoneType = 'campo';
    } else if (row === SAFE_ZONE_ROW) {
        zoneType = 'segura';
    } else if (row > SAFE_ZONE_ROW && row < BLUE_START_ROW) {
        zoneType = 'campo';
    } else {
        return null;
    }
    
    // Usar la función unificada para obtener el contenido
    const zoneInfo = getZoneInfoByType(zoneType);
    if (zoneInfo) {
        return {
            zone: zoneType,
            content: zoneInfo.content
        };
    }
    
    return null;
}

// Función para ocultar información de la zona
function hideZoneInfo() {
    const zoneInfoElement = document.getElementById('zoneInfo');
    if (!zoneInfoElement) return;
    
    // Mostrar mensaje por defecto centrado
    zoneInfoElement.innerHTML = `
        <p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>
    `;
    
    // Quitar pestaña activa
    const tabs = document.querySelectorAll('.tutorial-tab');
    tabs.forEach(tab => tab.classList.remove('active'));
}

// Función para resaltar zona desde pestaña
function highlightZoneFromTab(zone) {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar resaltado anterior
    tutorialBoard.querySelectorAll('.board-row').forEach(rowElement => {
        rowElement.classList.remove('zone-highlighted', 'field-zone', 'meta-highlighted');
    });
    
    // Determinar qué filas resaltar según la zona
    let rowsToHighlight = [];
    
    switch(zone) {
        case 'meta':
            rowsToHighlight = [0, 10]; // Meta azul y roja
            break;
        case 'aparicion':
            rowsToHighlight = [1, 9]; // Aparición roja y azul
            break;
        case 'campo':
            rowsToHighlight = [1, 2, 3, 4, 6, 7, 8, 9]; // Campos de ambos equipos + sus zonas de aparición
            break;
        case 'segura':
            rowsToHighlight = [5]; // Zona segura
            break;
    }
    
    // Resaltar las filas
    rowsToHighlight.forEach(rowIndex => {
        const rowElement = tutorialBoard.querySelector(`.board-row[data-row="${rowIndex}"]`);
        if (rowElement) {
            if (zone === 'meta') {
                // Para metas, usar clase específica sin contornos
                rowElement.classList.add('meta-highlighted');
            } else {
                // Para otras zonas, usar clase normal con contornos
                rowElement.classList.add('zone-highlighted');
                
                // Agregar clase field-zone para campos
                if (zone === 'campo') {
                    rowElement.classList.add('field-zone');
                }
            }
        }
    });
    
    // Mostrar fichas si es necesario
    if (zone === 'salida') {
        const zoneInfo = getZoneInfoByType(zone);
        if (zoneInfo && zoneInfo.showPieces && zoneInfo.pieces) {
            showTutorialPieces(zoneInfo.pieces);
        }
    } else {
        // Limpiar fichas para otras zonas
        clearTutorialPieces();
    }
    
    // Mostrar información de la zona
    const zoneInfo = getZoneInfoByType(zone);
    if (zoneInfo) {
        const zoneInfoElement = document.getElementById('zoneInfo');
        if (zoneInfoElement) {
            // Si hay selección de equipos, mostrar solo el contenido sin ejemplos
            if (zone === 'campo' || zone === 'meta' || zone === 'aparicion') {
                const contentWithoutExamples = zoneInfo.content.replace(/<div class="zone-examples">[\s\S]*?<\/div>/g, '');
                zoneInfoElement.innerHTML = contentWithoutExamples;
            } else {
                zoneInfoElement.innerHTML = zoneInfo.content;
            }
        }
    }
    
    // Mostrar/ocultar botones de equipo según la zona
    const teamSelection = document.getElementById('teamSelection');
    if (teamSelection) {
        if (zone === 'campo' || zone === 'meta' || zone === 'aparicion') {
            teamSelection.style.display = 'block';
            // Reiniciar subselección de equipos al cambiar de pestaña
            const teamFields = document.querySelectorAll('.team-field');
            teamFields.forEach(field => field.classList.remove('active'));
            // Actualizar textos según la zona
            updateTeamButtonTexts(zone);
        } else {
            teamSelection.style.display = 'none';
        }
    }
    
    // Actualizar la pestaña activa
    updateActiveTab(zone);
}

// Función para obtener información por tipo de zona
function getZoneInfoByType(zone) {
    const zoneInfo = {
        meta: {
            content: `
                <h3>Meta</h3>
                <p>El objetivo principal del juego es llegar a la meta con tus fichas</p>
                <p>Cada jugador debe llegar a la meta del lado contrario del tablero</p>
            `
        },
        aparicion: {
            content: `
                <h3>Aparición</h3>
                <p>Zona de aparición de las fichas con una formación aleatoria</p>
                <p>Las fichas tienen más libertad de movimiento en esta fila</p>
            `
        },
        campo: {
            content: `
                <h3>Campos de Equipo</h3>
                <p>Solo se puede eliminar fichas del equipo contrario en campo propio</p>
                <p>Ten cuidado al entrar en el campo contrario, podrías ser eliminado</p>
            `
        },
        segura: {
            content: `
                <h3>Zona Segura</h3>
                <p>Área neutral donde las fichas están seguras</p>
                <p>Una vez entres al seguro, nadie puede sacarte de allí</p>
                <p>La zona segura siempre está en el centro del tablero</p>
            `
        }
    };
    
    return zoneInfo[zone] || null;
}

// Inicializar event listeners para las pestañas
function initializeTutorialTabs() {
    const tabs = document.querySelectorAll('.tutorial-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Si la pestaña ya está activa, deseleccionarla
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                // Limpiar resaltado y mostrar mensaje por defecto
                clearTutorialHighlight();
            } else {
                // Actualizar pestaña activa
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Resaltar zona en el tablero
                highlightZoneFromTab(tab.dataset.zone);
            }
        });
    });
    
    // Asegurar que no hay pestañas activas al inicializar
    tabs.forEach(tab => tab.classList.remove('active'));
    
    // Inicializar botones de equipo
    initializeTeamButtons();
    
    // Mostrar mensaje por defecto
    hideZoneInfo();
}

// Función para actualizar textos de botones según la zona
function updateTeamButtonTexts(zone) {
    const teamTexts = {
        meta: {
            blue: "Debes llevar tus fichas a esta meta sin ser eliminado",
            red: "Evita que el rival llegue a esta meta con sus fichas"
        },
        aparicion: {
            blue: "Aquí aparecen tus fichas y tendrán mayor alcance",
            red: "El rival tendrá más rango de movimiento en esta fila"
        },
        campo: {
            blue: "En esta zona puedes eliminar fichas del rival",
            red: "Ten cuidado, el rival puede eliminarte en esta zona"
        }
    };
    
    const texts = teamTexts[zone];
    if (texts) {
        const blueField = document.querySelector('.team-field[data-team="blue"] .team-description');
        const redField = document.querySelector('.team-field[data-team="red"] .team-description');
        
        if (blueField) blueField.textContent = texts.blue;
        if (redField) redField.textContent = texts.red;
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para inicializar campos de equipo
function initializeTeamButtons() {
    const teamFields = document.querySelectorAll('.team-field');
    teamFields.forEach(field => {
        field.addEventListener('click', () => {
            // Si el campo ya está activo, deseleccionarlo
            if (field.classList.contains('active')) {
                field.classList.remove('active');
                // Volver al estado de la pestaña activa (mostrar ambas zonas)
                const activeTab = document.querySelector('.tutorial-tab.active');
                if (activeTab) {
                    highlightZoneFromTab(activeTab.dataset.zone);
                }
            } else {
                // Remover activo de todos los campos de equipo
                teamFields.forEach(f => f.classList.remove('active'));
                // Activar el campo clickeado
                field.classList.add('active');
                
                // Resaltar zona del equipo seleccionado
                highlightTeamField(field.dataset.team);
            }
        });
    });
}

// Función para resaltar zona de equipo específico
function highlightTeamField(team) {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Obtener la zona actual activa
    const activeTab = document.querySelector('.tutorial-tab.active');
    const currentZone = activeTab ? activeTab.dataset.zone : null;
    
    // Limpiar resaltado anterior
    tutorialBoard.querySelectorAll('.board-row').forEach(rowElement => {
        rowElement.classList.remove('zone-highlighted', 'field-zone', 'meta-highlighted');
    });
    
    let rowsToHighlight = [];
    let useMetaHighlight = false;
    
    if (currentZone === 'meta') {
        // Meta específica del equipo
        if (team === 'blue') {
            rowsToHighlight = [0]; // Meta azul
        } else if (team === 'red') {
            rowsToHighlight = [10]; // Meta roja
        }
        useMetaHighlight = true;
    } else if (currentZone === 'aparicion') {
        // Aparición específica del equipo
        if (team === 'blue') {
            rowsToHighlight = [9]; // Aparición azul
        } else if (team === 'red') {
            rowsToHighlight = [1]; // Aparición roja
        }
    } else if (currentZone === 'campo') {
        // Campo específico del equipo
        if (team === 'blue') {
            rowsToHighlight = [6, 7, 8, 9]; // Campo azul + aparición
        } else if (team === 'red') {
            rowsToHighlight = [1, 2, 3, 4]; // Campo rojo + aparición
        }
    }
    
    // Resaltar las filas
    rowsToHighlight.forEach(rowIndex => {
        const rowElement = tutorialBoard.querySelector(`.board-row[data-row="${rowIndex}"]`);
        if (rowElement) {
            if (useMetaHighlight) {
                rowElement.classList.add('meta-highlighted');
            } else {
                rowElement.classList.add('zone-highlighted', 'field-zone');
            }
        }
    });
}

// Función para limpiar resaltado del tutorial
function clearTutorialHighlight() {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar resaltado del tablero
    tutorialBoard.querySelectorAll('.board-row').forEach(rowElement => {
        rowElement.classList.remove('zone-highlighted', 'field-zone', 'meta-highlighted');
    });
    
    // Limpiar campos de equipo activos
    const teamFields = document.querySelectorAll('.team-field');
    teamFields.forEach(field => field.classList.remove('active'));
    
    // Ocultar selección de equipos
    const teamSelection = document.getElementById('teamSelection');
    if (teamSelection) {
        teamSelection.style.display = 'none';
    }
    
    // Limpiar fichas del tablero
    clearTutorialPieces();
    
    // Mostrar mensaje por defecto
    hideZoneInfo();
}

// Función para mostrar fichas en el tutorial
function showTutorialPieces(pieces) {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    // Limpiar fichas existentes
    clearTutorialPieces();
    
    // Añadir fichas
    pieces.forEach(piece => {
        const rowElement = tutorialBoard.querySelector(`.board-row[data-row="${piece.row}"]`);
        if (rowElement) {
            const cellElement = rowElement.querySelector(`.board-cell[data-col="${piece.col}"]`);
            if (cellElement) {
                const pieceElement = document.createElement('div');
                pieceElement.className = `piece ${piece.team}-piece`;
                pieceElement.style.width = '100%';
                pieceElement.style.height = '100%';
                pieceElement.style.borderRadius = '50%';
                pieceElement.style.backgroundColor = piece.team === 'red' ? '#E74C3C' : '#3498DB';
                pieceElement.style.border = '2px solid #ffffff';
                pieceElement.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.3)';
                cellElement.appendChild(pieceElement);
            }
        }
    });
}

// Función para limpiar fichas del tutorial
function clearTutorialPieces() {
    const tutorialBoard = document.getElementById('tutorialBoard');
    if (!tutorialBoard) return;
    
    const pieces = tutorialBoard.querySelectorAll('.piece');
    pieces.forEach(piece => piece.remove());
}

// Función para inicializar las pestañas de movimientos (completamente independiente)
function initializeMovementTabs() {
    const tabs = document.querySelectorAll('.movement-tab');
    const movementInfoElement = document.getElementById('movementInfo');
    
    if (!movementInfoElement) return;
    
    // Mostrar mensaje por defecto
    movementInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
    
    // Agregar event listeners a las pestañas
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Si la pestaña ya está activa, deseleccionarla
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                // Mostrar mensaje por defecto
                movementInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                // Actualizar pestaña activa
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Mostrar contenido de la zona
                const movement = tab.getAttribute('data-movement');
                showMovementInfo(movement);
            }
        });
    });
}

// Función para mostrar información de movimientos
function showMovementInfo(movement) {
    const movementInfoElement = document.getElementById('movementInfo');
    if (!movementInfoElement) return;
    
    const movementData = getMovementInfoByType(movement);
    if (movementData) {
        movementInfoElement.innerHTML = movementData.content;
    } else {
        movementInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para obtener información de movimientos
function getMovementInfoByType(movement) {
    const movementData = {
        salida: {
            content: `
                <h3>Movimientos de Salida</h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-movimientos-salida-${document.body.classList.contains('dark-theme') ? 'oscuro' : 'claro'}.png" alt="Movimientos de Salida" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Movimientos hacia delante o hacia los lados</p>
                        <p>En esta fila puedes moverte una o dos casillas</p>
                        <p>Si sales de la zona de aparición, no podrás volver a ella</p>
                    </div>
                </div>
            `
        },
        defensivo: {
            content: `
                <h3>Movimientos en Campo Propio</h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-movimientos-defensivo-${document.body.classList.contains('dark-theme') ? 'oscuro' : 'claro'}.png" alt="Campo Defensivo" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Movimientos hacia delante o hacia los lados</p>
                        <p>Solo puedes moverte una casilla en cada posible dirección</p>
                        <p>Podrás moverte así hasta que llegues a la zona segura</p>
                    </div>
                </div>
            `
        },
        ofensivo: {
            content: `
                <h3>Movimientos en Campo Contrario</h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-movimientos-ofensivo-${document.body.classList.contains('dark-theme') ? 'oscuro' : 'claro'}.png" alt="Campo Ofensivo" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Movimientos hacia delante; recto o en diagonal</p>
                        <p>En la zona de ataque, solo puedes moverte una casilla</p>
                        <p>Solamente puedes avanzar estando en el campo contrario</p>
                    </div>
                </div>
            `
        },
        meta: {
            content: `
                <h3>Movimiento para llegar a la meta</h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-movimientos-meta-${document.body.classList.contains('dark-theme') ? 'oscuro' : 'claro'}.png" alt="Zona de Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cuando llegues a la zona de aparición del rival, solo podrás entrar a la meta y sumar puntos</p>
                    </div>
                </div>
            `
        }
    };
    
    return movementData[movement] || null;
}

// Función para actualizar las imágenes de movimientos según el tema
function updateMovementImages() {
    const isDarkTheme = document.body.classList.contains('dark-theme');
    const themeSuffix = isDarkTheme ? 'oscuro' : 'claro';
    
    // Actualizar todas las imágenes de movimientos
    const images = document.querySelectorAll('.movement-image');
    images.forEach(img => {
        const src = img.src;
        if (src.includes('tutorial-movimientos-')) {
            const baseName = src.split('-').slice(0, -1).join('-');
            const newSrc = `${baseName}-${themeSuffix}.png`;
            img.src = newSrc;
        }
    });
    
    // Actualizar contenido si hay una pestaña activa
    const activeTab = document.querySelector('.movement-tab.active');
    if (activeTab) {
        const movement = activeTab.getAttribute('data-movement');
        showMovementInfo(movement);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para actualizar las imágenes cuando cambie el tema
function updateMovementImagesTheme() {
    // Solo actualizar si estamos en el paso de movimientos
    if (currentTutorialStep === 3) {
        updateMovementImages();
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}


// Función para obtener las filas que pertenecen a una zona
function getRowsForZone(row) {
    const BOARD_ROWS = 11;
    const BLUE_GOAL_ROW = 0;
    const RED_GOAL_ROW = BOARD_ROWS - 1;
    const RED_START_ROW = 1;
    const BLUE_START_ROW = BOARD_ROWS - 2;
    const SAFE_ZONE_ROW = Math.floor(BOARD_ROWS / 2);
    
    if (row === BLUE_GOAL_ROW || row === RED_GOAL_ROW) {
        // Metas - sin resaltado
        return [];
    } else if (row === RED_START_ROW || row === BLUE_START_ROW) {
        // Zonas de aparición - ambas apariciones (roja y azul)
        return [RED_START_ROW, BLUE_START_ROW];
    } else if (row < SAFE_ZONE_ROW && row > RED_START_ROW) {
        // Campo del equipo rojo - incluye zona de aparición (filas 1, 2, 3, 4)
        return [1, 2, 3, 4];
    } else if (row === SAFE_ZONE_ROW) {
        // Zona segura - solo la fila 5
        return [SAFE_ZONE_ROW];
    } else if (row > SAFE_ZONE_ROW && row < BLUE_START_ROW) {
        // Campo del equipo azul - incluye zona de aparición (filas 6, 7, 8, 9)
        return [6, 7, 8, 9];
    }
    
    return [row]; // Por defecto, solo la fila actual
}

// Función para obtener información de cada zona

// Función para crear el tablero Bala independiente
function createIndependentBalaBoard() {
    const previewBoard = document.getElementById('previewBoard');
    if (!previewBoard) return;
    
    // Limpiar contenido anterior
    previewBoard.innerHTML = '';
    
    // Configuración del tablero Bala (bala)
    const rows = 7;
    const cols = 5;
    const cellSize = 55; // Tamaño original de celda
    
    // Configurar el contenedor del tablero
    previewBoard.className = 'board bala-preview';
    previewBoard.style.display = 'flex';
    previewBoard.style.flexDirection = 'column';
    previewBoard.style.gap = '0';
    previewBoard.style.background = 'var(--board-bg)';
    previewBoard.style.padding = '20px';
    previewBoard.style.borderRadius = '12px';
    previewBoard.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
    previewBoard.style.border = '3px solid var(--board-border)';
    previewBoard.style.position = 'relative';
    previewBoard.style.transform = ''; // No aplicar transform aquí, se hace en CSS
    previewBoard.style.transformOrigin = 'top center';
    previewBoard.style.pointerEvents = 'none'; // Desactivar interacciones
    
    // Calcular dimensiones de las filas de meta
    const goalRowWidth = cellSize * cols;
    const goalRowHeight = 40;
    
    // Crear las filas del tablero
    for (let row = 0; row < rows; row++) {
        const boardRow = document.createElement('div');
        boardRow.className = 'board-row';
        boardRow.style.display = 'flex';
        boardRow.style.gap = '0';
        
        // Determinar el tipo de fila
        let rowType = 'normal';
        if (row === 0) {
            rowType = 'blue-goal';
        } else if (row === rows - 1) {
            rowType = 'red-goal';
        }
        
        if (rowType === 'blue-goal' || rowType === 'red-goal') {
            // Crear fila de meta
            const goalCell = document.createElement('div');
            goalCell.className = `board-cell ${rowType} goal-row`;
            goalCell.style.width = `${goalRowWidth}px`;
            goalCell.style.height = `${goalRowHeight}px`;
            goalCell.style.display = 'flex';
            goalCell.style.alignItems = 'center';
            goalCell.style.justifyContent = 'center';
            goalCell.style.border = 'none';
            goalCell.style.cursor = 'pointer';
            goalCell.style.pointerEvents = 'auto';
            
            // Aplicar estilos de meta
            if (rowType === 'blue-goal') {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-blue-primary) 0deg 90deg,
                        var(--goal-blue-secondary) 90deg 180deg,
                        var(--goal-blue-primary) 180deg 270deg,
                        var(--goal-blue-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            } else {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-red-primary) 0deg 90deg,
                        var(--goal-red-secondary) 90deg 180deg,
                        var(--goal-red-primary) 180deg 270deg,
                        var(--goal-red-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            }
            
            // Event listeners desactivados - el tablero no es responsivo al cursor
            // goalCell.addEventListener('mouseenter', () => highlightZone(row));
            // goalCell.addEventListener('mouseleave', () => clearHighlight());
            
            boardRow.appendChild(goalCell);
        } else {
            // Crear filas normales
            for (let col = 0; col < cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.cursor = 'default';
                cell.style.pointerEvents = 'none';
                cell.style.borderRight = '1px solid var(--board-grid-line)';
                cell.style.borderBottom = '1px solid var(--board-grid-line)';
                
                // Determinar el tipo de celda
                let cellType = 'neutral';
                if (row === 1) {
                    cellType = 'blue-start';
                } else if (row === rows - 2) {
                    cellType = 'red-start';
                } else if (row === 3) {
                    cellType = 'safe-zone'; // Toda la fila central es zona segura
                }
                
                cell.classList.add(cellType);
                
                // Aplicar estilos según el tipo
                if (cellType === 'neutral' || cellType === 'neutral2') {
                    cell.style.background = 'var(--cell-normal)';
                } else if (cellType === 'blue-start' || cellType === 'red-start') {
                    cell.style.background = 'var(--cell-start)';
                } else if (cellType === 'safe-zone') {
                    cell.style.background = 'var(--cell-safe)';
                    cell.style.position = 'relative';
                    
                    // Crear el indicador interno de zona segura
                    const innerZone = document.createElement('div');
                    innerZone.style.position = 'absolute';
                    innerZone.style.top = '4px';
                    innerZone.style.left = '4px';
                    innerZone.style.right = '4px';
                    innerZone.style.bottom = '4px';
                    innerZone.style.background = 'var(--cell-safe-inner)';
                    innerZone.style.borderRadius = '4px';
                    innerZone.style.opacity = '0.6';
                    innerZone.style.zIndex = '1';
                    cell.appendChild(innerZone);
                }
                
                // Remover bordes de los últimos elementos
                if (col === cols - 1) {
                    cell.style.borderRight = 'none';
                }
                
                boardRow.appendChild(cell);
            }
        }
        
        previewBoard.appendChild(boardRow);
    }
    
    // Remover el borde inferior de la última fila
    const lastRow = previewBoard.lastElementChild;
    if (lastRow && lastRow.lastElementChild) {
        lastRow.lastElementChild.style.borderBottom = 'none';
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para crear el tablero Marathon independiente
function createIndependentMarathonBoard() {
    const previewBoard = document.getElementById('previewBoard');
    if (!previewBoard) return;
    
    // Limpiar contenido anterior
    previewBoard.innerHTML = '';
    
    // Configuración del tablero Marathon (marathon)
    const rows = 15;
    const cols = 11;
    const cellSize = 55; // Tamaño original de celda
    
    // Configurar el contenedor del tablero
    previewBoard.className = 'board marathon-preview';
    previewBoard.style.display = 'flex';
    previewBoard.style.flexDirection = 'column';
    previewBoard.style.gap = '0';
    previewBoard.style.background = 'var(--board-bg)';
    previewBoard.style.padding = '20px';
    previewBoard.style.borderRadius = '12px';
    previewBoard.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.3)';
    previewBoard.style.border = '3px solid var(--board-border)';
    previewBoard.style.position = 'relative';
    previewBoard.style.transform = ''; // No aplicar transform aquí, se hace en CSS
    previewBoard.style.transformOrigin = 'top center';
    previewBoard.style.pointerEvents = 'none'; // Desactivar interacciones
    
    // Calcular dimensiones de las filas de meta
    const goalRowWidth = cellSize * cols;
    const goalRowHeight = 40;
    
    // Crear las filas del tablero
    for (let row = 0; row < rows; row++) {
        const boardRow = document.createElement('div');
        boardRow.className = 'board-row';
        boardRow.style.display = 'flex';
        boardRow.style.gap = '0';
        
        // Determinar el tipo de fila
        let rowType = 'normal';
        if (row === 0) {
            rowType = 'blue-goal';
        } else if (row === rows - 1) {
            rowType = 'red-goal';
        }
        
        if (rowType === 'blue-goal' || rowType === 'red-goal') {
            // Crear fila de meta
            const goalCell = document.createElement('div');
            goalCell.className = `board-cell ${rowType} goal-row`;
            goalCell.style.width = `${goalRowWidth}px`;
            goalCell.style.height = `${goalRowHeight}px`;
            goalCell.style.display = 'flex';
            goalCell.style.alignItems = 'center';
            goalCell.style.justifyContent = 'center';
            goalCell.style.border = 'none';
            goalCell.style.cursor = 'pointer';
            goalCell.style.pointerEvents = 'auto';
            
            // Aplicar estilos de meta
            if (rowType === 'blue-goal') {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-blue-primary) 0deg 90deg,
                        var(--goal-blue-secondary) 90deg 180deg,
                        var(--goal-blue-primary) 180deg 270deg,
                        var(--goal-blue-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            } else {
                goalCell.style.backgroundImage = `
                    repeating-conic-gradient(
                        var(--goal-red-primary) 0deg 90deg,
                        var(--goal-red-secondary) 90deg 180deg,
                        var(--goal-red-primary) 180deg 270deg,
                        var(--goal-red-secondary) 270deg 360deg
                    )
                `;
                goalCell.style.backgroundSize = '16px 16px';
            }
            
            // Event listeners desactivados - el tablero no es responsivo al cursor
            // goalCell.addEventListener('mouseenter', () => highlightZone(row));
            // goalCell.addEventListener('mouseleave', () => clearHighlight());
            
            boardRow.appendChild(goalCell);
        } else {
            // Crear filas normales
            for (let col = 0; col < cols; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.style.width = `${cellSize}px`;
                cell.style.height = `${cellSize}px`;
                cell.style.display = 'flex';
                cell.style.alignItems = 'center';
                cell.style.justifyContent = 'center';
                cell.style.cursor = 'default';
                cell.style.pointerEvents = 'none';
                cell.style.borderRight = '1px solid var(--board-grid-line)';
                cell.style.borderBottom = '1px solid var(--board-grid-line)';
                
                // Determinar el tipo de celda
                let cellType = 'neutral';
                if (row === 1) {
                    cellType = 'blue-start';
                } else if (row === rows - 2) {
                    cellType = 'red-start';
                } else if (row === 7) {
                    cellType = 'safe-zone'; // Toda la fila central (fila 7) es zona segura
                }
                
                cell.classList.add(cellType);
                
                // Aplicar estilos según el tipo
                if (cellType === 'neutral' || cellType === 'neutral2') {
                    cell.style.background = 'var(--cell-normal)';
                } else if (cellType === 'blue-start' || cellType === 'red-start') {
                    cell.style.background = 'var(--cell-start)';
                } else if (cellType === 'safe-zone') {
                    cell.style.background = 'var(--cell-safe)';
                    cell.style.position = 'relative';
                    
                    // Crear el indicador interno de zona segura
                    const innerZone = document.createElement('div');
                    innerZone.style.position = 'absolute';
                    innerZone.style.top = '4px';
                    innerZone.style.left = '4px';
                    innerZone.style.right = '4px';
                    innerZone.style.bottom = '4px';
                    innerZone.style.background = 'var(--cell-safe-inner)';
                    innerZone.style.borderRadius = '4px';
                    innerZone.style.border = '2px solid var(--cell-safe-border)';
                    innerZone.style.pointerEvents = 'none';
                    
                    cell.appendChild(innerZone);
                }
                
                boardRow.appendChild(cell);
            }
        }
        
        previewBoard.appendChild(boardRow);
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}

// Función para cerrar el tutorial interactivo
function closeInteractiveTutorial() {
    const interactiveTutorialScreen = document.getElementById('interactiveTutorialScreen');
    if (interactiveTutorialScreen) {
        interactiveTutorialScreen.classList.add('hidden');
    }
    
    // Limpiar el estado del tutorial
    currentTutorialStep = 1;
    updateTutorialStep();
    
    // Mostrar el menú de selección de tutorial
    const tutorialScreen = document.getElementById('tutorialScreen');
    if (tutorialScreen) {
        tutorialScreen.classList.remove('hidden');
    }
}

// Función para inicializar las pestañas de puntuación
function initializePuntuacionTabs() {
    const tabs = document.querySelectorAll('#step4 .movement-tab');
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');

    if (!puntuacionInfoElement) return;

    puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            if (tab.classList.contains('active')) {
                tab.classList.remove('active');
                puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center; background: none; border: none; padding: 0; margin: 0;">Selecciona una pestaña para obtener más información</p>';
            } else {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const puntuacion = tab.getAttribute('data-movement');
                showPuntuacionInfo(puntuacion);
            }
        });
    });
}

// Función para mostrar información de puntuación
function showPuntuacionInfo(puntuacion) {
    const puntuacionInfoElement = document.getElementById('puntuacionInfo');
    if (!puntuacionInfoElement) return;

    const puntuacionData = getPuntuacionInfoByType(puntuacion);
    if (puntuacionData) {
        puntuacionInfoElement.innerHTML = puntuacionData.content;
    } else {
        puntuacionInfoElement.innerHTML = '<p style="color: #cccccc; font-style: italic; text-align: center;">Información no disponible</p>';
    }
}

// Función para obtener información de puntuación por tipo
function getPuntuacionInfoByType(puntuacion) {
    const puntuacionData = {
        meta: {
            content: `
                <h3>Puntuación por llegar a la meta: <span style="color: #FFD700;">2 puntos</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-meta.png" alt="Meta" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Cruzar el campo es arriesgado, pero es la mejor forma de obtener puntos para ganar la partida</p>
                        <p>Evita que el rival llegue a la meta eliminando sus fichas</p>
                    </div>
                </div>
            `
        },
        eliminacion: {
            content: `
                <h3>Puntuación por Eliminación: <span style="color: #FFD700;">1 punto</span></h3>
                <div class="movement-layout">
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill2.png" alt="Eliminación 2" class="movement-image">
                    </div>
                    <div class="movement-image-container">
                        <img src="icons/tutorial-puntos-kill1.png" alt="Eliminación 1" class="movement-image">
                    </div>
                    <div class="movement-description">
                        <p>Elimina fichas del equipo contrario en tu campo</p>
                        <p>Si consigues dejar al rival sin fichas en el tablero, automáticamente ganarás la partida</p>
                    </div>
                </div>
            `
        }
    };
    return puntuacionData[puntuacion] || null;
}