import React, { useState } from 'react';
import { View, StyleSheet } from 'react-native';
import Board from '../components/Board';
import Controls from '../components/Controls';
import { BOARD_HEIGHT, BOARD_WIDTH } from '../constants/config';

export default function GameScreen() {
  // Estado del tablero: matriz 8x14 inicial
  const initialBoard = () => {
    const board = Array.from({ length: BOARD_HEIGHT }, () =>
      Array.from({ length: BOARD_WIDTH }, () => 0)
    );

    // Jugador 1 (parte inferior)
    for (let row = BOARD_HEIGHT - 2; row < BOARD_HEIGHT; row++) {
      for (let col = 0; col < BOARD_WIDTH; col++) {
        board[row][col] = 1;
      }
    }

    // Jugador 2 (parte superior)
    for (let row = 0; row < 2; row++) {
      for (let col = 0; col < BOARD_WIDTH; col++) {
        board[row][col] = 2;
      }
    }

    return board;
  };

  const [boardState, setBoardState] = useState(initialBoard);
  const [turn, setTurn] = useState(1);

  const resetGame = () => {
    setBoardState(initialBoard);
    setTurn(1);
  };

  const nextTurn = () => {
    setTurn((prev) => (prev === 1 ? 2 : 1));
  };

  return (
    <View style={styles.container}>
      <Board boardState={boardState} />
      <Controls currentPlayer={turn} onNextTurn={nextTurn} onReset={resetGame} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    paddingTop: 40,
    backgroundColor: '#fff',
    alignItems: 'center',
  },
});
