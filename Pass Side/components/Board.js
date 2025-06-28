import React from 'react';
import { View, StyleSheet } from 'react-native';
import Tile from './Tile';
import Piece from './Piece';
import { BOARD_WIDTH, BOARD_HEIGHT, SAFE_ROW } from '../constants/config';

export default function Board({ boardState }) {
  return (
    <View style={styles.board}>
      {Array.from({ length: BOARD_HEIGHT }).map((_, row) => (
        <View key={row} style={styles.row}>
          {Array.from({ length: BOARD_WIDTH }).map((_, col) => {
            const piece = boardState[row]?.[col];

            return (
              <Tile
                key={col}
                row={row}
                col={col}
                isSafe={row === SAFE_ROW}
              >
                {piece ? <Piece player={piece} /> : null}
              </Tile>
            );
          })}
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  board: {
    flexDirection: 'column',
    alignItems: 'center',
    marginTop: 20,
  },
  row: {
    flexDirection: 'row',
  },
});
