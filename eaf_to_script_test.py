from eaf_to_script import write_script, merge_turn_list, merge_turn_pair

def test_merge_turn_pair():
    turn1 = {'start': 0, 'end': 2, 'text': 'foo', 'speaker': 'Mr. Foo'}
    turn2 = {'start': 3, 'end': 4, 'text': 'bar', 'speaker': 'Mr. Foo'}
    merged = merge_turn_pair(turn1, turn2)
    assert merged == {'start': 0, 'end': 4, 'text': 'foo bar', 'speaker': 'Mr. Foo'}

def test_merge_turn_list():
    turn_list = [
        {'start': 0, 'end': 2, 'text': 'foo', 'speaker': 'Mr. Foo'},
        {'start': 1, 'end': 2, 'text': 'bar', 'speaker': 'Mr. Bar'},
        {'start': 3, 'end': 4, 'text': 'bar', 'speaker': 'Mr. Bar'},
        {'start': 5, 'end': 6, 'text': 'baz', 'speaker': 'Mr. Baz'},
    ]
    merged = merge_turn_list(turn_list)
    assert merged == [
        {'start': 0, 'end': 2, 'text': 'foo', 'speaker': 'Mr. Foo'},
        {'start': 1, 'end': 4, 'text': 'bar bar', 'speaker': 'Mr. Bar'},
        {'start': 5, 'end': 6, 'text': 'baz', 'speaker': 'Mr. Baz'},
    ]

def test_merge_turn_list1():
    turn_list = [
        {'start': 0,  'end': 2,  'text': 'foo', 'speaker': 'Mr. Foo'},
        {'start': 1,  'end': 2,  'text': 'bar', 'speaker': 'Mr. Bar'},
        {'start': 3,  'end': 4,  'text': 'bar', 'speaker': 'Mr. Bar'},
        {'start': 5,  'end': 6,  'text': 'baz', 'speaker': 'Mr. Baz'},
        {'start': 7,  'end': 8,  'text': 'baz', 'speaker': 'Mr. Baz'},
        {'start': 9,  'end': 9,  'text': 'baz', 'speaker': 'Mr. Baz'},
        {'start': 10, 'end': 11, 'text': 'baz', 'speaker': 'Mr. Baz'},
    ]
    merged = merge_turn_list(turn_list)
    assert merged == [
        {'start': 0, 'end': 2,  'text': 'foo',              'speaker': 'Mr. Foo'},
        {'start': 1, 'end': 4,  'text': 'bar bar',          'speaker': 'Mr. Bar'},
        {'start': 5, 'end': 11, 'text': 'baz baz baz baz',  'speaker': 'Mr. Baz'},
    ]